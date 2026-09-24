import Foundation

/// Configuration for a CoreML draft model used in speculative decoding.
///
/// The draft model runs on ANE (Apple Neural Engine) via CoreML and predicts
/// a short sequence of candidate tokens that the main Metal model then verifies
/// in parallel. A good draft model shares the same tokenizer family as the main
/// model and is small enough that ANE inference is near-instantaneous.
public struct DraftModelConfig: Sendable, Equatable {
    /// URL to the compiled CoreML `.mlpackage` or `.mlmodelc` directory.
    public let modelURL: URL

    /// Number of candidate tokens the draft model predicts per speculative
    /// round. Higher K increases throughput when acceptance rate is high but
    /// wastes more work on rejection. Typical range: 3–8.
    public let draftDepth: Int

    /// Sampling temperature for the draft model. A value of 0.0 (greedy)
    /// maximises acceptance rate against a greedy main-model decode.
    public let temperature: Float

    /// Optional top-k truncation applied to the draft model's logits before
    /// sampling. `nil` means no truncation.
    public let topK: Int?

    /// Whether to prefer ANE placement. When `true` the CoreML configuration
    /// requests `.all` compute units with ANE priority. When `false` the draft
    /// model falls back to GPU-only (useful for debugging).
    public let preferANE: Bool

    /// Maximum sequence length (in tokens) the draft model accepts as context.
    /// Must be >= the main model's context window used during speculative rounds.
    public let maxContextLength: Int

    /// Acceptance strategy for verifying draft tokens against the main model.
    public let acceptanceStrategy: AcceptanceStrategy

    /// Whether to dynamically adjust draft depth (K) based on rolling
    /// acceptance rate. When `true`, K increases when acceptance is high
    /// and decreases when it drops, staying within 1...10.
    public let enableAdaptiveK: Bool

    /// Acceptance rate below which speculative decoding is disabled for the
    /// remainder of the generation. `nil` means never disable.
    public let fallbackThreshold: Float?

    public init(
        modelURL: URL,
        draftDepth: Int = 5,
        temperature: Float = 0.0,
        topK: Int? = nil,
        preferANE: Bool = true,
        maxContextLength: Int = 4096,
        acceptanceStrategy: AcceptanceStrategy = .tokenMatch,
        enableAdaptiveK: Bool = true,
        fallbackThreshold: Float? = 0.15
    ) {
        precondition((1...10).contains(draftDepth),
                     "draftDepth must be between 1 and 10, got \(draftDepth)")
        precondition(temperature >= 0, "temperature must be non-negative")
        if let topK {
            precondition((1...256).contains(topK),
                         "topK must be between 1 and 256, got \(topK)")
        }
        self.modelURL = modelURL
        self.draftDepth = draftDepth
        self.temperature = temperature
        self.topK = topK
        self.preferANE = preferANE
        self.maxContextLength = maxContextLength
        self.acceptanceStrategy = acceptanceStrategy
        self.enableAdaptiveK = enableAdaptiveK
        self.fallbackThreshold = fallbackThreshold
    }

    /// Production default for Gemma 2B drafting Gemma 26B-A4B.
    public static func gemma2BDefault(modelURL: URL) -> DraftModelConfig {
        DraftModelConfig(
            modelURL: modelURL,
            draftDepth: 5,
            temperature: 0.0,
            topK: nil,
            preferANE: true,
            maxContextLength: 4096,
            acceptanceStrategy: .tokenMatch,
            enableAdaptiveK: true,
            fallbackThreshold: 0.15
        )
    }
}

/// Errors specific to draft model loading and inference.
public enum DraftModelError: Error, CustomStringConvertible, Sendable {
    case modelNotFound(url: URL)
    case modelLoadFailed(url: URL, underlyingError: Error)
    case invalidInputShape(expected: [Int], actual: [Int])
    case inferenceFailed(underlyingError: Error)
    case missingOutputFeature(name: String)

    public var description: String {
        switch self {
        case .modelNotFound(let url):
            return "draft model not found at \(url.path)"
        case .modelLoadFailed(let url, let err):
            return "failed to load draft model from \(url.path): \(err)"
        case .invalidInputShape(let expected, let actual):
            return "invalid input shape: expected \(expected), got \(actual)"
        case .inferenceFailed(let err):
            return "draft model inference failed: \(err)"
        case .missingOutputFeature(let name):
            return "draft model output missing feature '\(name)'"
        }
    }
}
