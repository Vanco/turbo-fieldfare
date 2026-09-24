import Foundation
import CoreML

/// CoreML-backed engine for running a small stateful draft model on ANE.
///
/// Handles the Gemma 4 E2B CoreML model which uses the MLState API for
/// internal KV cache management.
///
/// Usage pattern:
/// 1. `warmContext(tokens)` — replay tokens into the KV cache (once per round)
/// 2. `predictNextWithInput(token:)` — single-step forward, returns next token
/// 3. `resetState()` — clear KV cache (after rejection or new generation)
public final class CoreMLDraftEngine: @unchecked Sendable {
    private let model: MLModel
    private var state: MLState
    private let config: DraftModelConfig
    private let queue: DispatchQueue

    private let inputIDsName = "input_ids"
    private let positionIDsName = "position_ids"
    private let updateMaskName = "update_mask"
    private let causalMaskName = "causal_mask"
    private let outputTokenName = "token_id"
    private let outputLogitName = "token_logit"

    public let contextWindow: Int
    public let vocabSize: Int
    private var kvPosition: Int = 0

    public let isANEAvailable: Bool

    public init(config: DraftModelConfig) throws {
        let fm = FileManager.default
        guard fm.fileExists(atPath: config.modelURL.path) else {
            throw DraftModelError.modelNotFound(url: config.modelURL)
        }

        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = config.preferANE ? .all : .cpuAndGPU

        let model: MLModel
        do {
            model = try MLModel(contentsOf: config.modelURL, configuration: mlConfig)
        } catch {
            throw DraftModelError.modelLoadFailed(url: config.modelURL,
                                                  underlyingError: error)
        }

        self.model = model
        self.config = config
        self.state = model.makeState()
        self.queue = DispatchQueue(label: "com.turbofieldfare.draft-engine",
                                   qos: .userInitiated)

        let desc = model.modelDescription
        if let maskDesc = desc.inputDescriptionsByName[updateMaskName],
           let multiArray = maskDesc.multiArrayConstraint,
           multiArray.shape.count >= 3 {
            self.contextWindow = multiArray.shape[2].intValue
        } else {
            self.contextWindow = config.maxContextLength
        }

        self.vocabSize = 262_144
        self.isANEAvailable = config.preferANE
    }

    /// Reset the KV cache by creating a fresh MLState.
    public func resetState() {
        queue.sync { resetStateInternal() }
    }

    public var position: Int {
        queue.sync { kvPosition }
    }

    /// Replay tokens into the KV cache. Call once per speculative round.
    public func warmContext(tokens: [Int32]) throws {
        try queue.sync {
            resetStateInternal()
            for token in tokens {
                _ = try runStep(token: token, position: kvPosition)
                kvPosition += 1
            }
        }
    }

    /// Run a single forward step: feed `token` at the current KV position
    /// and return the model's prediction for the next position.
    public func predictNextWithInput(token: Int32) throws -> (tokenID: Int32, logit: Float) {
        try queue.sync {
            let pos = kvPosition
            let result = try runStepWithLogit(token: token, position: pos)
            kvPosition = pos + 1
            return result
        }
    }

    // MARK: - Private

    private func resetStateInternal() {
        state = model.makeState()
        kvPosition = 0
    }

    private func runStep(token: Int32, position: Int) throws -> Int32 {
        try runStepWithLogit(token: token, position: position).tokenID
    }

    private func runStepWithLogit(token: Int32, position: Int) throws -> (tokenID: Int32, logit: Float) {
        let C = contextWindow

        let inputIDs = try MLMultiArray(shape: [1, 1], dataType: .int32)
        inputIDs[[0, 0]] = NSNumber(value: token)

        let positionIDs = try MLMultiArray(shape: [1], dataType: .int32)
        positionIDs[0] = NSNumber(value: position)

        let updateMask = try MLMultiArray(shape: [1, 1, NSNumber(value: C), 1], dataType: .float16)
        updateMask.dataPointer.assumingMemoryBound(to: Float16.self)[position] = Float16(1.0)

        let causalMask = try MLMultiArray(shape: [1, 1, 1, NSNumber(value: C)], dataType: .float16)
        let causalPtr = causalMask.dataPointer.assumingMemoryBound(to: Float16.self)
        let cs = causalMask.strides[3].intValue
        let one: Float16 = Float16(1.0)
        for i in 0...min(position, C - 1) {
            causalPtr[cs * i] = one
        }

        let provider = try MLDictionaryFeatureProvider(dictionary: [
            inputIDsName: MLFeatureValue(multiArray: inputIDs),
            positionIDsName: MLFeatureValue(multiArray: positionIDs),
            updateMaskName: MLFeatureValue(multiArray: updateMask),
            causalMaskName: MLFeatureValue(multiArray: causalMask),
        ])

        let output: MLFeatureProvider
        do {
            output = try model.prediction(from: provider, using: state)
        } catch {
            throw DraftModelError.inferenceFailed(underlyingError: error)
        }

        guard let tokenValue = output.featureValue(for: outputTokenName),
              let tokenArray = tokenValue.multiArrayValue else {
            throw DraftModelError.missingOutputFeature(name: outputTokenName)
        }
        let tokenID = Int32(truncating: tokenArray[0])

        var logit: Float = 0
        if let logitValue = output.featureValue(for: outputLogitName),
           let logitArray = logitValue.multiArrayValue {
            logit = Float(truncating: logitArray[0])
        }

        return (tokenID, logit)
    }
}
