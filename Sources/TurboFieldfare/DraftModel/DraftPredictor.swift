import Foundation

/// Autoregressive K-token predictor backed by a CoreML draft engine.
///
/// Uses the stateful engine's incremental API:
/// 1. `warmContext` replays the prompt into the KV cache (once)
/// 2. `predictNextWithInput` feeds one token and returns the next prediction
///
/// This gives O(K) inference cost per round instead of O(K²).
public final class DraftPredictor: @unchecked Sendable {
    private let engine: CoreMLDraftEngine
    let config: DraftModelConfig
    private let queue: DispatchQueue

    private var totalDrafted: Int = 0
    private var totalAccepted: Int = 0

    public var acceptanceRate: Float {
        guard totalDrafted > 0 else { return 0 }
        return Float(totalAccepted) / Float(totalDrafted)
    }

    public init(engine: CoreMLDraftEngine, config: DraftModelConfig) {
        self.engine = engine
        self.config = config
        self.queue = DispatchQueue(label: "com.turbofieldfare.draft-predictor",
                                   qos: .userInitiated)
    }

    /// Predict `k` candidate tokens from the given context.
    public func predictKTokens(context: [Int32], k: Int) throws -> [Int32] {
        // Warm the KV cache with the full context.
        try engine.warmContext(tokens: context)

        var candidates: [Int32] = []
        candidates.reserveCapacity(k)
        var lastToken = context[context.count - 1]

        for _ in 0..<k {
            let (token, _) = try engine.predictNextWithInput(token: lastToken)
            candidates.append(token)
            lastToken = token
        }

        return candidates
    }

    /// Predict `k` candidate tokens with their logit values.
    public func predictKTokensWithLogits(context: [Int32], k: Int) throws -> (
        tokens: [Int32],
        logits: [[Float]]
    ) {
        try engine.warmContext(tokens: context)

        var candidates: [Int32] = []
        var logitsHistory: [[Float]] = []
        candidates.reserveCapacity(k)
        logitsHistory.reserveCapacity(k)
        var lastToken = context[context.count - 1]

        for _ in 0..<k {
            let (token, logit) = try engine.predictNextWithInput(token: lastToken)
            candidates.append(token)
            logitsHistory.append([logit])
            lastToken = token
        }

        return (candidates, logitsHistory)
    }

    public func recordRoundOutcome(acceptedCount: Int, draftedCount: Int) {
        queue.sync {
            totalDrafted += draftedCount
            totalAccepted += acceptedCount
        }
    }

    public func resetStatistics() {
        queue.sync {
            totalDrafted = 0
            totalAccepted = 0
        }
    }

    public func adaptiveDraftDepth() -> Int {
        let rate = acceptanceRate
        switch rate {
        case 0.85...:
            return min(config.draftDepth + 2, 10)
        case 0.70..<0.85:
            return config.draftDepth
        case 0.50..<0.70:
            return max(config.draftDepth - 1, 1)
        default:
            return 1
        }
    }
}
