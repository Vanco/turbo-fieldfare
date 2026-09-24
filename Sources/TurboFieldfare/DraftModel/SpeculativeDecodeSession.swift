import Foundation
import Metal

/// Result of a single speculative decoding round.
public struct SpeculativeRoundResult: Sendable {
    /// Tokens accepted by the main model (may be empty on full rejection).
    public let acceptedTokens: [Int32]
    /// The bonus token sampled from the main model when all draft tokens
    /// matched, or the corrected token at the first rejection point.
    /// `nil` if the round produced no usable token (e.g., max context hit).
    public let bonusToken: Int32?
    /// Number of draft tokens that matched the main model's output.
    public let acceptedCount: Int
    /// Total number of draft tokens proposed.
    public let draftCount: Int
    /// Whether the generation should stop (EOS, stop string, max tokens).
    public let shouldStop: Bool
    /// Stop reason if `shouldStop` is true.
    public let stopReason: StopReason?

    public init(acceptedTokens: [Int32],
                bonusToken: Int32?,
                acceptedCount: Int,
                draftCount: Int,
                shouldStop: Bool,
                stopReason: StopReason? = nil) {
        self.acceptedTokens = acceptedTokens
        self.bonusToken = bonusToken
        self.acceptedCount = acceptedCount
        self.draftCount = draftCount
        self.shouldStop = shouldStop
        self.stopReason = stopReason
    }

    /// Total new tokens produced in this round (accepted + bonus).
    public var totalNewTokens: Int {
        acceptedTokens.count + (bonusToken != nil ? 1 : 0)
    }
}

/// Orchestrates speculative decoding between a draft model (ANE via CoreML)
/// and the main Metal model.
///
/// Each round:
/// 1. Draft model predicts K candidate tokens (fast, on ANE)
/// 2. Main model verifies each candidate sequentially via `produce()`
/// 3. Accepted tokens are committed; on rejection, KV rewinds and corrects
/// 4. A bonus token is sampled if all K candidates matched
///
/// The session manages `DraftKVContext` to keep the draft model's input
/// consistent with the main model's KV state.
public final class SpeculativeDecodeSession: @unchecked Sendable {
    let predictor: DraftPredictor
    private let draftContext: DraftKVContext
    private let kvSync: KVCacheSync
    private let logitsBuffer: MTLBuffer
    private let probsBuffer: MTLBuffer
    private let outTokenBuffer: MTLBuffer
    private let sampler: Sampler
    private let vocab: Int
    let acceptanceTest: AcceptanceTest

    /// Cumulative statistics across all rounds.
    public private(set) var totalDrafted: Int = 0
    public private(set) var totalAccepted: Int = 0
    public private(set) var totalRounds: Int = 0

    /// Whether speculative decoding is currently active. Set to `false`
    /// when the acceptance rate falls below the fallback threshold.
    public private(set) var isActive: Bool = true

    /// Wall-clock time spent in the draft model (ANE) across all rounds.
    public private(set) var draftModelSeconds: Double = 0
    /// Wall-clock time spent in the main model (Metal) across all rounds.
    public private(set) var mainModelSeconds: Double = 0

    /// Acceptance rate across all rounds (accepted / drafted).
    public var acceptanceRate: Float {
        guard totalDrafted > 0 else { return 0 }
        return Float(totalAccepted) / Float(totalDrafted)
    }

    /// Average tokens generated per round (the "speedup factor").
    public var averageTokensPerRound: Float {
        guard totalRounds > 0 else { return 0 }
        return Float(totalAccepted) / Float(totalRounds)
    }

    init(predictor: DraftPredictor,
         draftContext: DraftKVContext,
         context: MetalContext,
         vocab: Int = 262_144) throws {
        self.predictor = predictor
        self.draftContext = draftContext
        self.kvSync = KVCacheSync(draftContext: draftContext)
        self.vocab = vocab
        self.acceptanceTest = AcceptanceTest(
            strategy: predictor.config.acceptanceStrategy,
            draftDepth: predictor.config.draftDepth)

        guard let logits = context.device.makeBuffer(
            length: vocab * MemoryLayout<Float16>.size,
            options: .storageModeShared),
              let probs = context.device.makeBuffer(
            length: vocab * MemoryLayout<Float16>.size,
            options: .storageModeShared),
              let outToken = context.device.makeBuffer(
            length: MemoryLayout<UInt32>.size,
            options: .storageModeShared)
        else {
            throw ModelError.residentBufferWrapFailed
        }
        self.logitsBuffer = logits
        self.probsBuffer = probs
        self.outTokenBuffer = outToken
        self.sampler = try Sampler(context: context, vocab: vocab)
    }

    /// Seed the logits buffer from an external source (e.g., prefill output).
    ///
    /// After prefill, the main model's logits for the first decode position
    /// are in `scratch.logits`. This method copies them into the session's
    /// buffer so the first speculative verification checks against the
    /// correct logits.
    public func seedLogits(from source: MTLBuffer) {
        let bytes = vocab * MemoryLayout<Float16>.size
        logitsBuffer.contents().copyMemory(from: source.contents(), byteCount: bytes)
    }

    /// Public factory: create a speculative decode session from a config URL
    /// without exposing internal types to callers.
    public static func make(
        config: DraftModelConfig,
        context: MetalContext,
        maxContextLength: Int,
        vocab: Int = 262_144
    ) throws -> SpeculativeDecodeSession {
        let engine = try CoreMLDraftEngine(config: config)
        let predictor = DraftPredictor(engine: engine, config: config)
        let draftCtx = DraftKVContext(maxContextLength: maxContextLength)
        return try SpeculativeDecodeSession(
            predictor: predictor,
            draftContext: draftCtx,
            context: context,
            vocab: vocab)
    }

    /// Execute one speculative decoding round.
    ///
    /// - Parameters:
    ///   - runner: The main Metal forward runner. Must have its KV cursor at
    ///     the position matching `draftContext.position`.
    ///   - config: Generation config (temperature, top-k, etc.).
    ///   - context: Metal context for GPU command submission.
    ///   - stopTokenIDs: Token IDs that terminate generation.
    ///   - position: Current KV position (must equal `draftContext.position`).
    /// - Returns: The round result with accepted tokens and optional bonus.
    public func runRound(
        runner: RealForwardRunner,
        config: GenerationConfig,
        context: MetalContext,
        stopTokenIDs: Set<Int32>,
        position: Int
    ) async throws -> SpeculativeRoundResult {
        precondition(draftContext.position == position,
                     "draft context position \(draftContext.position) != runner position \(position)")

        // Snapshot for rollback on failure.
        kvSync.snapshot()

        // Adaptive K: use the rolling acceptance rate to choose draft depth.
        let K: Int
        if predictor.config.enableAdaptiveK {
            K = predictor.adaptiveDraftDepth()
        } else {
            K = predictor.config.draftDepth
        }

        // Step 1: Draft model proposes K tokens, timed.
        let draftStart = CFAbsoluteTimeGetCurrent()
        let draftResult = try predictor.predictKTokensWithLogits(
            context: draftContext.inputContext, k: K)
        let draftTokens = draftResult.tokens
        draftModelSeconds += CFAbsoluteTimeGetCurrent() - draftStart

        totalRounds += 1
        totalDrafted += K

        // Step 2: Verify each draft token through the main model.
        //
        // Pre-condition: logitsBuffer already contains the main model's
        // logits predicting `position` (seeded from prefill or the last
        // produce() call of the previous round).  We check each draft
        // token against those logits BEFORE feeding it, which is the
        // correct order for speculative decoding verification.
        var acceptedTokens: [Int32] = []
        var bonusToken: Int32?
        var shouldStop = false
        var stopReason: StopReason?
        var currentPosition = position
        var rejectedCorrectedToken: Int32?

        let verifyStart = CFAbsoluteTimeGetCurrent()

        for (i, draftToken) in draftTokens.prefix(K).enumerated() {
            // Step A: Check draft token against logits that predict
            // `currentPosition` (what the main model thinks belongs here).
            let mainLogitsPtr = logitsBuffer.contents()
                .assumingMemoryBound(to: Float16.self)
            let logitsBuf = UnsafeBufferPointer(start: mainLogitsPtr, count: vocab)

            let acceptanceResult = acceptanceTest.test(
                draftToken: draftToken,
                mainLogits: logitsBuf,
                vocab: vocab,
                config: config)

            if acceptanceResult.accepted {
                acceptedTokens.append(draftToken)

                // Step B: Feed accepted token — advance KV, get logits for
                // the next position (used by the next iteration's check).
                // Use produceWithLogits to ensure the full logit vector is
                // written (not the fused greedy head which skips logits).
                try await runner.produceWithLogits(
                    token: draftToken,
                    position: currentPosition,
                    into: logitsBuffer)

                kvSync.advanceMain()
                currentPosition += 1

                if stopTokenIDs.contains(draftToken) {
                    shouldStop = true
                    stopReason = resolveStopReason(draftToken, config: config)
                    break
                }
            } else {
                // Step C: Rejected — feed the main model's own token to
                // advance KV.  No rewind needed because we checked BEFORE
                // producing.
                rejectedCorrectedToken = acceptanceResult.mainToken

                try await runner.produceWithLogits(
                    token: acceptanceResult.mainToken,
                    position: currentPosition,
                    into: logitsBuffer)

                kvSync.advanceMain()
                currentPosition += 1

                if stopTokenIDs.contains(acceptanceResult.mainToken) {
                    shouldStop = true
                    stopReason = resolveStopReason(acceptanceResult.mainToken, config: config)
                } else {
                    bonusToken = acceptanceResult.mainToken
                }
                break
            }
        }

        mainModelSeconds += CFAbsoluteTimeGetCurrent() - verifyStart

        let acceptedCount = acceptedTokens.count

        // Step 3: If all K draft tokens matched, sample a bonus token.
        if acceptedCount == K && !shouldStop {
            if let seed = try? sampleFromLogitsOptional(
                logits: logitsBuffer,
                config: config,
                context: context,
                position: currentPosition) {
                if stopTokenIDs.contains(seed) {
                    shouldStop = true
                    stopReason = resolveStopReason(seed, config: config)
                } else {
                    bonusToken = seed
                    // Feed the bonus token into the main model so its KV
                    // entry is written (otherwise the KV cursor falls behind
                    // mainPosition and the next round would mismatch).
                    try await runner.produceWithLogits(
                        token: seed,
                        position: currentPosition,
                        into: logitsBuffer)
                    // Advance main position for the bonus token so that
                    // commitAccepted validation passes (mainPosition must
                    // match basePosition + tokensToCommit.count).
                    kvSync.advanceMain()
                    currentPosition += 1
                }
            }
        }

        // Commit to draft context.
        if let correctedToken = rejectedCorrectedToken {
            // Rejection: append accepted tokens + corrected token.
            kvSync.commitRejection(
                acceptedTokens: acceptedTokens,
                correctedToken: correctedToken,
                basePosition: position)
        } else {
            // All accepted (or hit stop): batch-append accepted tokens.
            // If a bonus token was also produced, include it so the draft
            // context and mainPosition track the full KV advance.
            var tokensToCommit = acceptedTokens
            if let bonus = bonusToken {
                tokensToCommit.append(bonus)
            }
            kvSync.commitAccepted(tokens: tokensToCommit, basePosition: position)
        }

        totalAccepted += acceptedCount

        // Update predictor and acceptance test statistics.
        predictor.recordRoundOutcome(acceptedCount: acceptedCount, draftedCount: K)
        acceptanceTest.recordRound(acceptedInRound: acceptedCount)

        // Check fallback threshold: disable speculative decoding if the
        // rolling acceptance rate is too low.
        if let threshold = predictor.config.fallbackThreshold,
           acceptanceTest.totalTested >= 20,
           acceptanceTest.rollingAcceptanceRate < threshold {
            isActive = false
        }

        return SpeculativeRoundResult(
            acceptedTokens: acceptedTokens,
            bonusToken: bonusToken,
            acceptedCount: acceptedCount,
            draftCount: K,
            shouldStop: shouldStop,
            stopReason: stopReason)
    }

    /// Synchronise the draft model's context after a non-speculative event
    /// (e.g., prefill, continuation, or manual token injection).
    ///
    /// This replaces the draft context's token history to match the main
    /// model's KV state.
    public func syncContext(tokens: [Int32], position: Int) {
        kvSync.syncAfterPrefill(tokens: tokens, position: position)
    }

    /// Reset all state for a new generation.
    public func reset() {
        kvSync.reset()
        predictor.resetStatistics()
        acceptanceTest.reset()
        totalDrafted = 0
        totalAccepted = 0
        totalRounds = 0
        isActive = true
        draftModelSeconds = 0
        mainModelSeconds = 0
    }

    // MARK: - Private

    private func sampleFromLogits(
        logits: MTLBuffer,
        config: GenerationConfig,
        context: MetalContext,
        position: Int
    ) throws -> Int32 {
        let cb = context.queue.makeCommandBuffer()!
        sampler.sample(commandBuffer: cb,
                       logits: logits,
                       probs: probsBuffer,
                       history: [],
                       config: config,
                       position: position,
                       outToken: outTokenBuffer)
        cb.commit()
        cb.waitUntilCompleted()
        try checkCommandBufferError(cb)
        return Int32(bitPattern: outTokenBuffer.contents().load(as: UInt32.self))
    }

    private func sampleFromLogitsOptional(
        logits: MTLBuffer,
        config: GenerationConfig,
        context: MetalContext,
        position: Int
    ) throws -> Int32? {
        try sampleFromLogits(logits: logits, config: config,
                             context: context, position: position)
    }

    private func resolveStopReason(_ tokenID: Int32, config: GenerationConfig) -> StopReason {
        // Caller is responsible for checking EOS/EOT — this is a minimal
        // mapping for the speculative path. The real stop matching happens
        // in the outer decode loop (StreamingStopMatcher).
        .eos
    }
}
