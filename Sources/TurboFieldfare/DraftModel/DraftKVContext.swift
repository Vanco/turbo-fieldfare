import Foundation

/// Manages the token context for the draft model during speculative decoding.
///
/// The draft model (CoreML on ANE) receives a token sequence as input and
/// maintains its own internal KV cache. This context tracker provides the
/// correct input window at each speculative step and supports truncation
/// when the main model rejects draft tokens.
///
/// The context mirrors the main model's KV position: after each accepted
/// token the context advances, and after a rejection it rewinds to the
/// last accepted position so the draft model sees consistent input.
final class DraftKVContext: @unchecked Sendable {
    /// The full token history seen by the draft model (prompt + generated).
    private(set) var tokenIDs: [Int32]

    /// Current position (= number of tokens the draft model has processed).
    private(set) var position: Int

    /// Maximum context length the draft model accepts.
    let maxContextLength: Int

    /// Sliding window size for draft model input. If the context exceeds this,
    /// only the most recent `windowSize` tokens are fed to the draft model.
    /// `nil` means use the full context (up to `maxContextLength`).
    let windowSize: Int?

    init(initialTokens: [Int32] = [], maxContextLength: Int = 4096, windowSize: Int? = nil) {
        self.tokenIDs = initialTokens
        self.position = initialTokens.count
        self.maxContextLength = maxContextLength
        self.windowSize = windowSize
    }

    /// The context window to feed the draft model at the current position.
    ///
    /// Returns either the full token history or a sliding window tail,
    /// truncated to `maxContextLength`.
    var inputContext: [Int32] {
        let available: [Int32]
        if let window = windowSize, tokenIDs.count > window {
            available = Array(tokenIDs.suffix(window))
        } else {
            available = tokenIDs
        }
        if available.count > maxContextLength {
            return Array(available.suffix(maxContextLength))
        }
        return available
    }

    /// Append an accepted token to the context.
    ///
    /// - Parameter token: The token ID accepted by the main model.
    func append(token: Int32) {
        tokenIDs.append(token)
        position += 1
    }

    /// Append multiple accepted tokens (from a successful speculative round).
    ///
    /// - Parameter tokens: Token IDs accepted in order.
    func append(contentsOf tokens: [Int32]) {
        tokenIDs.append(contentsOf: tokens)
        position += tokens.count
    }

    /// Truncate the context to a target position, discarding all tokens after it.
    ///
    /// Called when the main model rejects a draft token: rewind the context
    /// so the next speculative round starts from the correct position.
    ///
    /// - Parameter targetPosition: The position to rewind to (inclusive).
    ///   Must be ≤ current `position`.
    func rewind(to targetPosition: Int) {
        precondition(targetPosition >= 0,
                     "rewind target must be non-negative")
        precondition(targetPosition <= position,
                     "rewind target \(targetPosition) exceeds position \(position)")
        let discardCount = position - targetPosition
        if discardCount > 0 {
            tokenIDs.removeLast(discardCount)
        }
        position = targetPosition
    }

    /// Reset the context to empty (e.g., for a new conversation turn).
    func reset() {
        tokenIDs.removeAll(keepingCapacity: true)
        position = 0
    }

    /// Reset and replace with new initial tokens (e.g., after a prefill).
    func reset(to tokens: [Int32]) {
        tokenIDs = tokens
        position = tokens.count
    }
}
