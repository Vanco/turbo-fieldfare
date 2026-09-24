import Foundation

/// Coordinates KV cache state between the main Metal model and the draft
/// CoreML model during speculative decoding.
///
/// The main model's KV is managed by `KVCacheManager` (direct buffer access);
/// the draft model's KV is managed internally by CoreML. This coordinator
/// ensures both models see consistent token histories and provides batch
/// operations for applying accepted tokens and handling rejections.
///
/// ## Design
///
/// After each speculative round the coordinator:
/// 1. **Commits** accepted tokens to the draft context (batch append)
/// 2. **Truncates** on rejection (rewind to last accepted position)
/// 3. **Snapshots** the context state for rollback on failure
/// 4. **Validates** that draft and main model positions stay in sync
public final class KVCacheSync: @unchecked Sendable {
    /// The draft model's token context (feeds CoreML input).
    private let draftContext: DraftKVContext

    /// Position the main model's KV cache currently holds.
    /// Updated after each successful `produce()` call.
    public private(set) var mainPosition: Int

    /// Position the draft model's context currently holds.
    public var draftPosition: Int { draftContext.position }

    /// Whether the two positions are in sync.
    public var isSynchronized: Bool { mainPosition == draftPosition }

    /// Maximum rewind safety margin (tokens).
    private let maxRewindMargin: Int

    /// Snapshot stack for rollback on generation failure.
    private var snapshots: [(mainPosition: Int, draftTokens: [Int32])] = []
    private let maxSnapshots = 10

    init(draftContext: DraftKVContext, initialPosition: Int = 0) {
        self.draftContext = draftContext
        self.mainPosition = initialPosition
        self.maxRewindMargin = 64
    }

    // MARK: - Position Tracking

    /// Update the main model's position after a successful `produce()` call.
    public func advanceMain(by count: Int = 1) {
        mainPosition += count
    }

    // MARK: - Batch Commit

    /// Apply a batch of accepted tokens to both draft and main contexts.
    ///
    /// Called after a speculative round where N tokens were accepted.
    /// The main model's KV already has K/V written for these tokens via
    /// sequential `produce()` calls; this only updates the position tracker
    /// and draft context.
    ///
    /// - Parameters:
    ///   - tokens: The accepted token IDs (in order).
    ///   - basePosition: The KV position before these tokens were written.
    public func commitAccepted(tokens: [Int32], basePosition: Int) {
        guard !tokens.isEmpty else { return }

        // Validate that main model position is consistent.
        let expectedMainPosition = basePosition + tokens.count
        precondition(mainPosition == expectedMainPosition,
                     "commitAccepted: main position \(mainPosition) != "
                         + "expected \(expectedMainPosition) "
                         + "(base \(basePosition) + \(tokens.count) tokens)")

        // Batch-append to draft context.
        draftContext.append(contentsOf: tokens)
    }

    /// Commit a round that ended in rejection: append the accepted prefix
    /// and the main model's corrected token to the draft context.
    ///
    /// In the new verification flow (check-then-produce), the rejected
    /// draft token was never fed to the main model, so no KV rewind is
    /// needed.  The corrected token was produced instead, advancing KV
    /// by one position.
    ///
    /// - Parameters:
    ///   - acceptedTokens: Draft tokens that were accepted before the rejection.
    ///   - correctedToken: The main model's own token used at the rejection point.
    ///   - basePosition: The KV position before this round started.
    public func commitRejection(
        acceptedTokens: [Int32],
        correctedToken: Int32,
        basePosition: Int
    ) {
        // Append accepted prefix to draft context.
        if !acceptedTokens.isEmpty {
            draftContext.append(contentsOf: acceptedTokens)
        }
        // Append the corrected token.
        draftContext.append(token: correctedToken)
        // Update position: base + accepted + 1 (corrected token).
        mainPosition = basePosition + acceptedTokens.count + 1
    }

    // MARK: - Rejection Rollback

    /// Handle a draft token rejection: rewind both contexts to the position
    /// just before the rejected token, then apply the corrected token.
    ///
    /// - Parameters:
    ///   - rejectionPosition: The position where the draft token was rejected.
    ///   - correctedToken: The main model's token to use instead.
    public func handleRejection(rejectionPosition: Int, correctedToken: Int32) {
        // Rewind the draft context to the rejection point.
        draftContext.rewind(to: rejectionPosition)

        // Apply the corrected token.
        draftContext.append(token: correctedToken)

        // Main model's KV was already rewound and re-produced by the caller.
        // Update position to reflect the corrected token.
        mainPosition = rejectionPosition + 1
    }

    // MARK: - Context Synchronization

    /// Full context sync after prefill or continuation.
    ///
    /// Replaces the draft model's token history to match the main model's
    /// KV state exactly.
    ///
    /// - Parameters:
    ///   - tokens: The complete token history the main model holds.
    ///   - position: The main model's KV position.
    public func syncAfterPrefill(tokens: [Int32], position: Int) {
        draftContext.reset(to: tokens)
        mainPosition = position

        precondition(isSynchronized,
                     "syncAfterPrefill: positions out of sync after sync "
                         + "(main=\(mainPosition), draft=\(draftPosition))")
    }

    // MARK: - Snapshot / Restore

    /// Save the current context state for potential rollback.
    ///
    /// Called at the start of each speculative round so that a generation
    /// failure can be recovered gracefully.
    public func snapshot() {
        let snap = (mainPosition: mainPosition,
                    draftTokens: draftContext.tokenIDs)
        snapshots.append(snap)
        if snapshots.count > maxSnapshots {
            snapshots.removeFirst()
        }
    }

    /// Restore the most recent snapshot. Returns the restored position.
    ///
    /// Used for error recovery: if a speculative round fails mid-way,
    /// restore to the snapshot and retry or fall back to standard decode.
    @discardableResult
    public func restoreSnapshot() -> Int {
        guard let snap = snapshots.popLast() else {
            return mainPosition
        }
        draftContext.reset(to: snap.draftTokens)
        mainPosition = snap.mainPosition
        return mainPosition
    }

    /// Discard all snapshots (e.g., after a successful generation completes).
    public func clearSnapshots() {
        snapshots.removeAll(keepingCapacity: true)
    }

    // MARK: - Validation

    /// Validate that the context is in a consistent state.
    ///
    /// - Returns: `nil` if consistent, error message if not.
    public func validateConsistency() -> String? {
        if mainPosition != draftPosition {
            return "position mismatch: main=\(mainPosition), draft=\(draftPosition)"
        }
        if draftContext.tokenIDs.count != draftPosition {
            return "draft token count \(draftContext.tokenIDs.count) "
                + "!= position \(draftPosition)"
        }
        return nil
    }

    // MARK: - Reset

    /// Reset all state for a new generation.
    public func reset() {
        draftContext.reset()
        mainPosition = 0
        snapshots.removeAll(keepingCapacity: true)
    }
}
