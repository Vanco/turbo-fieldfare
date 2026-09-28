import Foundation
import TurboFieldfare

/// One prefill chunk's own share of the model-cumulative expert counters.
///
/// The events carry running totals, so a chunk's cost only exists as the
/// difference between two of them. `ioRatio` is that chunk's read time over
/// its wall time: the read timer is summed across the threads that serve
/// routed experts, so a ratio at or above 1 means the reads were fully
/// serialized on the chunk's critical path, which is what an I/O bound chunk
/// costs. It is a ratio, not a disk utilization figure — a healthy prefill on
/// this machine measures around 0.1, a cold expert pool measures above 3, and
/// a 2 tok/s collapse measures far higher.
struct PrefillIOChunk: Equatable {
    let wallNanos: UInt64
    let readNanos: UInt64
    let bytes: UInt64
    let misses: Int
    let hits: Int

    var ioRatio: Double {
        wallNanos == 0 ? 0 : Double(readNanos) / Double(wallNanos)
    }

    var isBlocking: Bool { ioRatio >= PrefillIOTracker.blockingRatio }
}

struct PrefillIOSummary: Equatable {
    let chunks: Int
    let blockedChunks: Int
    let worstRatio: Double
    let bytesRead: UInt64
    let blockedWallNanos: UInt64
    let totalWallNanos: UInt64

    /// Share of the prefill's wall time spent in chunks whose reads outran the
    /// chunk. This is the line between a cold pool and a stalled prefill, and it
    /// is deliberately a share of time rather than a count of chunks: faulting
    /// a 12 GB pool into the cache blocks the first three chunks every time and
    /// costs about 7% of a 247s prefill, which is a healthy prefill that paid a
    /// one-off cost. The 09-27 collapse was 77% — thousands of chunks' worth of
    /// the same wait, repeated. Counting chunks alone cannot tell those apart.
    var blockedShare: Double {
        totalWallNanos == 0 ? 0 : Double(blockedWallNanos) / Double(totalWallNanos)
    }
}

/// Turns consecutive cumulative prefill events into per-chunk deltas and
/// remembers the worst chunk, so one call at the end of the prefill can say
/// whether the whole run was I/O bound.
///
/// Per request, because the counters are per model and the deltas are only
/// meaningful within a single prefill: the first chunk of a later request has
/// no meaningful delta against the last chunk of an earlier one.
final class PrefillIOTracker: @unchecked Sendable {
    /// A chunk whose expert reads consumed at least its whole wall time. A
    /// round number on purpose: it is the point where read time meets wall
    /// time, so it needs no tuning against a particular machine's disk.
    static let blockingRatio = 1.0

    private let lock = NSLock()
    private let start: ContinuousClock.Instant
    private let now: @Sendable () -> ContinuousClock.Instant
    private var previous: ExpertIOStats?
    private var previousAt: ContinuousClock.Instant
    private var chunks = 0
    private var blockedChunks = 0
    private var worstRatio = 0.0
    private var lastBytes: UInt64 = 0
    private var blockedWallNanos: UInt64 = 0
    private var totalWallNanos: UInt64 = 0

    init(start: ContinuousClock.Instant = .now,
         now: @escaping @Sendable () -> ContinuousClock.Instant = { ContinuousClock.now }) {
        self.start = start
        self.now = now
        self.previousAt = start
    }

    func record(cumulative: ExpertIOStats, baseline: ExpertIOStats) -> PrefillIOChunk {
        let instant = now()
        lock.lock()
        defer { lock.unlock() }
        // The first chunk of a prefill is measured against the totals the
        // prefill started from, not against a previous chunk it never had.
        let before = previous ?? baseline
        let beforeAt = previous == nil ? start : previousAt
        previous = cumulative
        previousAt = instant
        lastBytes = cumulative.bytesRead
        let wallNanos = max(1, Self.nanos(beforeAt.duration(to: instant)))
        let chunk = PrefillIOChunk(
            wallNanos: UInt64(wallNanos),
            readNanos: cumulative.readNanos &- before.readNanos,
            bytes: cumulative.bytesRead &- before.bytesRead,
            misses: cumulative.missCount - before.missCount,
            hits: cumulative.hitCount - before.hitCount)
        chunks += 1
        totalWallNanos += chunk.wallNanos
        if chunk.isBlocking {
            blockedChunks += 1
            blockedWallNanos += chunk.wallNanos
        }
        worstRatio = max(worstRatio, chunk.ioRatio)
        return chunk
    }

    /// `Duration` keeps seconds and attoseconds in separate fields, so a
    /// chunk longer than a second needs both to report a real time. A chunk
    /// that rounds to zero is floored at one nanosecond, which would make its
    /// ratio look infinite rather than unknown.
    private static func nanos(_ duration: Duration) -> Int64 {
        let parts = duration.components
        let seconds = max(0, parts.seconds)
        let attoseconds = max(0, parts.attoseconds)
        return seconds * 1_000_000_000 + attoseconds / 1_000_000_000
    }

    /// `nil` unless a majority of the prefill's wall time went to chunks that
    /// were read-blocked. A prefill that fails before reporting cannot claim it
    /// was I/O bound, and neither can one that merely paid the cold-start cost.
    var summary: PrefillIOSummary? {
        lock.lock()
        defer { lock.unlock() }
        guard chunks > 0, blockedChunks > 0 else { return nil }
        let summary = PrefillIOSummary(chunks: chunks,
                                      blockedChunks: blockedChunks,
                                      worstRatio: worstRatio,
                                      bytesRead: lastBytes,
                                      blockedWallNanos: blockedWallNanos,
                                      totalWallNanos: totalWallNanos)
        return summary.blockedShare > 0.5 ? summary : nil
    }
}

enum ServerLog {
    static func accepted(id: String, streaming: Bool) {
        write("request \(id) accepted streaming=\(streaming)")
    }

    static func prefill(id: String,
                        actual: Int,
                        total: Int,
                        io: ExpertIOStats,
                        chunk: PrefillIOChunk) {
        // `io` is cumulative for the model, so these are running totals. The
        // hit rate is what separates a stall caused by expert I/O from one
        // caused by something else: a prefill that re-reads the same experts
        // every chunk holds a steady miss rate and a falling hit rate.
        //
        // `io=blocked` is the per-chunk verdict, and it is what makes the
        // difference actionable. Cumulative totals alone look the same at
        // every point of a stall, because a stall and a slow-but-healthy
        // prefill differ in rate, not in totals.
        write("request \(id) prefill \(actual)/\(total) "
            + "expert_gb=\(gib(io.bytesRead)) "
            + "read_s=\(seconds(io.readNanos)) "
            + "misses=\(io.missCount) "
            + "hits=\(io.hitCount) "
            + "read_mb_s=\(megabytesPerSecond(io)) "
            + "chunk_s=\(seconds(chunk.wallNanos)) "
            + "io_ratio=\(String(format: "%.2f", chunk.ioRatio)) "
            + "io=\(chunk.isBlocking ? "blocked" : "ok")")
    }

    private static func gib(_ bytes: UInt64) -> String {
        String(format: "%.2f", Double(bytes) / 1_073_741_824.0)
    }

    private static func seconds(_ nanos: UInt64) -> String {
        String(format: "%.1f", Double(nanos) / 1e9)
    }

    private static func megabytesPerSecond(_ io: ExpertIOStats) -> String {
        String(format: "%.0f", io.bytesPerSecond / 1_048_576.0)
    }

    static func prepared(id: String, promptTokens: Int?) {
        let count = promptTokens.map(String.init) ?? "backend-managed"
        write("request \(id) prepared prompt=\(count)")
    }

    static func queued(id: String) {
        write("request \(id) queued")
    }

    static func generating(id: String) {
        write("request \(id) generating")
    }

    /// Emitted once, after a prefill that spent a majority of its time in
    /// chunks blocked on expert reads. Without it the only trace of a stall is a
    /// quiet gap between two `prefill` lines, and nothing tells an operator that
    /// the wait was disk, not compute. Names the two knobs that change the read
    /// volume instead of restating the symptom.
    static func prefillIOBound(id: String, summary: PrefillIOSummary) {
        write(prefillIOBoundMessage(id: id, summary: summary))
    }

    /// Split out so the wording can be checked without a log sink, like
    /// `cancelledMessage`. Must name the two knobs that change read volume, so
    /// a reader does not have to already know which stage was slow.
    static func prefillIOBoundMessage(id: String, summary: PrefillIOSummary) -> String {
        "request \(id) prefill io-bound "
            + "blocked=\(String(format: "%.0f%%", summary.blockedShare * 100)) "
            + "blocked_chunks=\(summary.blockedChunks)/\(summary.chunks) "
            + "worst_ratio=\(String(format: "%.2f", summary.worstRatio)) "
            + "expert_gb=\(gib(summary.bytesRead)) "
            + "note=expert reads outran prefill compute; raise "
            + "--expert-cache-slots or free memory so the expert pool stays cached"
    }

    static func completed(id: String,
                          duration: Duration,
                          completion: ServerCompletion) {
        let usage = completion.usage
        write("request \(id) completed in \(format(duration)) "
            + "prompt=\(usage.promptTokens) "
            + "cached=\(usage.promptTokensDetails.cachedTokens) "
            + "completion=\(usage.completionTokens) "
            + "finish=\(completion.finishReason)")
    }

    /// The session is an actor, so generation is serialized: this line always
    /// belongs to the request between the preceding `generating` and the
    /// following `completed`. Carries a reason code only, never prompt content.

    static func visionPackInvalid(at url: URL, error: Error) {
        write("vision pack at \(url.path) is invalid: \(String(reflecting: error))")
    }

    static func visionRuntimeUnsupported(at url: URL, error: Error) {
        write("vision runtime for pack at \(url.path) is unsupported: "
            + String(reflecting: error))
    }

    static func visionRuntimeUnsupported() {
        write("vision runtime is unsupported: the image tower requires an M2 or newer Mac")
    }

    /// A prefix that could not be continued because its bridge failed to
    /// render. Carries the underlying error, because this miss means the
    /// template and the cached turn disagree, which no other miss does.
    static func promptCacheBridgeFailed(error: Error) {
        write("prompt cache miss reason=bridge-render-failed "
            + "error=\(String(reflecting: error))")
    }

    /// A request whose client stopped listening. Kept distinct from `failed`
    /// because a cancel is not a server fault and should not read as one, and
    /// added because excluding it from logging altogether left the request's last
    /// line as `generating` forever: an operator could not tell a running request
    /// from an abandoned one, or from a crashed one. Carries no content.
    static func cancelled(id: String, phase: String, duration: Duration) {
        write(cancelledMessage(id: id, phase: phase, duration: duration))
    }

    /// Split out so the wording can be checked without a log sink: this line
    /// must stay distinguishable from `failed`, must name the phase, and must
    /// carry no prompt or generated content.
    static func cancelledMessage(id: String, phase: String, duration: Duration) -> String {
        "request \(id) cancelled by client in \(format(duration)) phase=\(phase)"
    }

    static func failed(id: String,
                       phase: String,
                       status: UInt,
                       error: Error) {
        write("request \(id) failed phase=\(phase) status=\(status) "
            + "error=\(String(reflecting: error))")
    }

    private static func format(_ duration: Duration) -> String {
        let seconds = Double(duration.components.seconds)
            + Double(duration.components.attoseconds) / 1e18
        return String(format: "%.3fs", seconds)
    }

    private static func write(_ message: String) {
        let line = "[\(Date().formatted(.iso8601))] \(message)\n"
        FileHandle.standardError.write(Data(line.utf8))
    }
}
