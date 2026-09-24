import Foundation

/// Strategy for accepting or rejecting a draft token during speculative
/// decoding. Different strategies trade off acceptance rate against
/// generation quality.
public enum AcceptanceStrategy: Sendable, Equatable, CustomStringConvertible {
    /// Accept if the main model's sampled token exactly matches the draft
    /// token. Standard approach for greedy decode.
    case tokenMatch

    /// Accept if the draft token falls within the main model's top-k
    /// predictions. More permissive — accepts tokens the main model
    /// considered likely even if it sampled a different one.
    case topK(k: Int)

    /// Accept if the main model's probability for the draft token meets
    /// a minimum threshold. Uses the full logit distribution.
    /// `threshold` is a probability in (0, 1].
    case probabilityThreshold(threshold: Float)

    public var description: String {
        switch self {
        case .tokenMatch: return "token-match"
        case .topK(let k): return "top-k(\(k))"
        case .probabilityThreshold(let t): return "probability-threshold(\(t))"
        }
    }
}

/// Result of testing a single draft token against the main model's output.
public struct AcceptanceResult: Sendable {
    public let accepted: Bool
    public let mainToken: Int32
    public let strategy: AcceptanceStrategy
    /// Main model's probability for the draft token (0 if not computed).
    public let draftProbability: Float
    /// Main model's probability for its own chosen token.
    public let mainProbability: Float

    public init(accepted: Bool,
                mainToken: Int32,
                strategy: AcceptanceStrategy,
                draftProbability: Float = 0,
                mainProbability: Float = 0) {
        self.accepted = accepted
        self.mainToken = mainToken
        self.strategy = strategy
        self.draftProbability = draftProbability
        self.mainProbability = mainProbability
    }
}

/// Accumulates acceptance statistics and provides the core acceptance test
/// used by `SpeculativeDecodeSession`.
///
/// The acceptance test operates on logit buffers from the main model:
/// 1. Apply the configured acceptance strategy
/// 2. Record the outcome for statistics
public final class AcceptanceTest: @unchecked Sendable {
    public let strategy: AcceptanceStrategy

    // MARK: - Statistics

    public private(set) var totalTested: Int = 0
    public private(set) var totalAccepted: Int = 0
    public private(set) var totalRejected: Int = 0
    private var cumulativeDraftProb: Double = 0
    private var cumulativeAcceptedMainProb: Double = 0
    private var recentRoundAccepted: [Int] = []
    private let maxRecentRounds = 100
    private let predictorDraftDepth: Int

    /// Rolling acceptance rate over the last `maxRecentRounds` rounds.
    public var rollingAcceptanceRate: Float {
        guard !recentRoundAccepted.isEmpty else { return 0 }
        let total = recentRoundAccepted.reduce(0, +)
        return Float(total) / Float(recentRoundAccepted.count * max(1, predictorDraftDepth))
    }

    /// Overall acceptance rate.
    public var acceptanceRate: Float {
        guard totalTested > 0 else { return 0 }
        return Float(totalAccepted) / Float(totalTested)
    }

    /// Average probability the main model assigns to accepted draft tokens.
    public var averageAcceptedConfidence: Float {
        guard totalAccepted > 0 else { return 0 }
        return Float(cumulativeAcceptedMainProb / Double(totalAccepted))
    }

    public init(strategy: AcceptanceStrategy = .tokenMatch, draftDepth: Int = 5) {
        self.strategy = strategy
        self.predictorDraftDepth = draftDepth
    }

    public func reset() {
        totalTested = 0
        totalAccepted = 0
        totalRejected = 0
        cumulativeDraftProb = 0
        cumulativeAcceptedMainProb = 0
        recentRoundAccepted.removeAll(keepingCapacity: true)
    }

    // MARK: - Core Acceptance Test

    /// Test whether a draft token should be accepted given the main model's
    /// logits and the sampling configuration.
    public func test(
        draftToken: Int32,
        mainLogits: UnsafeBufferPointer<Float16>,
        vocab: Int,
        config: GenerationConfig
    ) -> AcceptanceResult {
        totalTested += 1

        let draftIdx = Int(draftToken)
        guard draftIdx >= 0, draftIdx < vocab else {
            totalRejected += 1
            let mainTok = argmaxLogits(mainLogits, vocab: vocab)
            return AcceptanceResult(accepted: false, mainToken: mainTok,
                                    strategy: strategy)
        }

        switch strategy {
        case .tokenMatch:
            return testTokenMatch(draftIdx: draftIdx, mainLogits: mainLogits,
                                  vocab: vocab, config: config)
        case .topK(let k):
            return testTopK(draftIdx: draftIdx, k: k, mainLogits: mainLogits,
                            vocab: vocab, config: config)
        case .probabilityThreshold(let threshold):
            return testProbabilityThreshold(
                draftIdx: draftIdx, threshold: threshold,
                mainLogits: mainLogits, vocab: vocab, config: config)
        }
    }

    /// Record a round's acceptance outcome for rolling statistics.
    public func recordRound(acceptedInRound: Int) {
        recentRoundAccepted.append(acceptedInRound)
        if recentRoundAccepted.count > maxRecentRounds {
            recentRoundAccepted.removeFirst()
        }
    }

    // MARK: - Strategy Implementations

    private func testTokenMatch(
        draftIdx: Int,
        mainLogits: UnsafeBufferPointer<Float16>,
        vocab: Int,
        config: GenerationConfig
    ) -> AcceptanceResult {
        let mainTok = argmaxLogits(mainLogits, vocab: vocab)
        let accepted = draftIdx == Int(mainTok)
        let draftProb = softmaxProbability(of: draftIdx, in: mainLogits, vocab: vocab)
        cumulativeDraftProb += Double(draftProb)

        if accepted {
            totalAccepted += 1
            cumulativeAcceptedMainProb += Double(draftProb)
        } else {
            totalRejected += 1
        }

        return AcceptanceResult(
            accepted: accepted, mainToken: mainTok, strategy: strategy,
            draftProbability: draftProb,
            mainProbability: accepted ? draftProb : 0)
    }

    private func testTopK(
        draftIdx: Int, k: Int,
        mainLogits: UnsafeBufferPointer<Float16>,
        vocab: Int,
        config: GenerationConfig
    ) -> AcceptanceResult {
        let mainTok = argmaxLogits(mainLogits, vocab: vocab)
        let draftProb = softmaxProbability(of: draftIdx, in: mainLogits, vocab: vocab)
        cumulativeDraftProb += Double(draftProb)

        let topKTokens = topKIndices(mainLogits, vocab: vocab, k: k)
        let inTopK = topKTokens.contains(draftIdx)
        let accepted = inTopK || draftIdx == Int(mainTok)

        if accepted {
            totalAccepted += 1
            cumulativeAcceptedMainProb += Double(draftProb)
        } else {
            totalRejected += 1
        }

        return AcceptanceResult(
            accepted: accepted, mainToken: mainTok, strategy: strategy,
            draftProbability: draftProb,
            mainProbability: accepted ? draftProb : 0)
    }

    private func testProbabilityThreshold(
        draftIdx: Int, threshold: Float,
        mainLogits: UnsafeBufferPointer<Float16>,
        vocab: Int,
        config: GenerationConfig
    ) -> AcceptanceResult {
        let mainTok = argmaxLogits(mainLogits, vocab: vocab)
        let draftProb = softmaxProbability(of: draftIdx, in: mainLogits, vocab: vocab)
        cumulativeDraftProb += Double(draftProb)

        let accepted = draftProb >= threshold || draftIdx == Int(mainTok)

        if accepted {
            totalAccepted += 1
            cumulativeAcceptedMainProb += Double(draftProb)
        } else {
            totalRejected += 1
        }

        return AcceptanceResult(
            accepted: accepted, mainToken: mainTok, strategy: strategy,
            draftProbability: draftProb,
            mainProbability: accepted ? draftProb : 0)
    }

    // MARK: - Utilities

    /// Argmax over FP16 logits (greedy token).
    func argmaxLogits(_ logits: UnsafeBufferPointer<Float16>, vocab: Int) -> Int32 {
        var bestIdx: Int = 0
        var bestVal: Float = -Float.infinity
        for i in 0..<vocab {
            let v = Float(logits[i])
            if v > bestVal {
                bestVal = v
                bestIdx = i
            }
        }
        return Int32(bestIdx)
    }

    /// Softmax probability of a specific token index.
    func softmaxProbability(
        of index: Int,
        in logits: UnsafeBufferPointer<Float16>,
        vocab: Int
    ) -> Float {
        var maxLogit: Float = -Float.infinity
        for i in 0..<vocab {
            let v = Float(logits[i])
            if v > maxLogit { maxLogit = v }
        }
        var sumExp: Float = 0
        var targetExp: Float = 0
        for i in 0..<vocab {
            let e = exp(Float(logits[i]) - maxLogit)
            sumExp += e
            if i == index { targetExp = e }
        }
        return targetExp / sumExp
    }

    /// Return the indices of the top-k logits (unsorted by magnitude).
    func topKIndices(
        _ logits: UnsafeBufferPointer<Float16>,
        vocab: Int,
        k: Int
    ) -> [Int] {
        guard k > 0, k <= vocab else { return Array(0..<vocab) }
        // Partial selection via simple scan — O(vocab) which is fine for
        // verification (vocab = 262K, k <= 256).
        var indexed: [(Int, Float)] = []
        indexed.reserveCapacity(vocab)
        for i in 0..<vocab {
            indexed.append((i, Float(logits[i])))
        }
        indexed.sort { $0.1 > $1.1 }
        return indexed.prefix(k).map(\.0)
    }
}
