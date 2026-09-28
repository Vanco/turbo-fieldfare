import Testing
import Foundation
import TurboFieldfare
@testable import TurboFieldfareServerCore

/// A prefill that stalls on the expert pool used to be indistinguishable from
/// one that is merely slow: both leave a quiet gap between two `prefill` lines,
/// because the counters are cumulative and a stall changes their rate, not their
/// totals. The tracker turns those totals into per-chunk deltas so the gap
/// itself carries a verdict.
@Suite struct ServerPrefillIOTests {
    /// A clock the test advances by hand, so chunk durations are exact rather
    /// than whatever the machine happened to be doing.
    private final class FakeClock: @unchecked Sendable {
        private let lock = NSLock()
        private let base = ContinuousClock.Instant.now
        private var elapsed: Duration = .zero

        var now: ContinuousClock.Instant {
            lock.lock(); defer { lock.unlock() }
            return base.advanced(by: elapsed)
        }

        func advance(by duration: Duration) {
            lock.lock(); defer { lock.unlock() }
            elapsed += duration
        }
    }

    @Test func aChunkReportsOnlyItsOwnDeltaRatherThanTheRunningTotal() {
        let clock = FakeClock()
        let tracker = PrefillIOTracker(start: clock.now, now: { clock.now })
        // Cumulative totals already include two chunks of work by the time the
        // second event lands, so the second chunk's delta must exclude the
        // first chunk's reads.
        tracker.record(cumulative: ExpertIOStats(bytesRead: 100, readNanos: 1_000_000_000,
                                                missCount: 4, hitCount: 10),
                       baseline: ExpertIOStats())
        clock.advance(by: .seconds(2))
        let second = tracker.record(cumulative: ExpertIOStats(bytesRead: 160, readNanos: 1_400_000_000,
                                                             missCount: 6, hitCount: 30),
                                    baseline: ExpertIOStats())

        #expect(second.bytes == 60)
        #expect(second.readNanos == 400_000_000)
        #expect(second.misses == 2)
        #expect(second.hits == 20)
    }

    @Test func aChunkShorterThanASecondReportsRealNanoseconds() {
        let clock = FakeClock()
        let tracker = PrefillIOTracker(start: clock.now, now: { clock.now })
        tracker.record(cumulative: ExpertIOStats(), baseline: ExpertIOStats())
        clock.advance(by: .milliseconds(400))
        let chunk = tracker.record(cumulative: ExpertIOStats(readNanos: 100_000_000),
                                   baseline: ExpertIOStats())

        // A chunk whose whole seconds field is zero still has to report 400ms,
        // or a fast chunk reads as an instant one and its ratio is meaningless.
        #expect(chunk.wallNanos == 400_000_000)
        #expect(abs(chunk.ioRatio - 0.25) < 1e-9)
    }

    @Test func aColdStartThatFaultedThePoolIsNotAdviceToActOn() {
        // Measured, not modelled: a 12,013-token prefill in 247s whose chunks
        // two and three blocked at ratios of 3.84 and 3.45 while faulting 12 GB
        // into the cache, then never blocked again. That is a healthy prefill
        // that paid a one-off cost, and an advisory here would send an operator
        // to change a setting that is already right. The per-chunk marks still
        // land; only the advisory is withheld.
        let clock = FakeClock()
        let tracker = PrefillIOTracker(start: clock.now, now: { clock.now })
        // (wall seconds, read seconds) per chunk, from /tmp/verify.log.
        let profile: [(wall: Double, read: Double)] = [
            (13.4, 1.9), (5.6, 21.4), (5.9, 20.5), (4.8, 0.3), (4.9, 0.3), (5.1, 0.4),
        ]
        var bytes: UInt64 = 0
        var nanos: UInt64 = 0
        for (index, chunk) in profile.enumerated() {
            bytes += 3_800_000_000
            nanos += UInt64(chunk.read * 1e9)
            clock.advance(by: .nanoseconds(Int64(chunk.wall * 1e9)))
            let recorded = tracker.record(
                cumulative: ExpertIOStats(bytesRead: bytes, readNanos: nanos),
                baseline: ExpertIOStats())
            if index == 1 || index == 2 {
                #expect(recorded.isBlocking, "chunk \(index) should be marked")
            } else {
                #expect(!recorded.isBlocking, "chunk \(index) should be clean")
            }
        }
        for _ in 0..<41 {
            bytes += 120_000_000
            nanos += 300_000_000
            clock.advance(by: .seconds(5))
            #expect(!tracker.record(cumulative: ExpertIOStats(bytesRead: bytes,
                                                            readNanos: nanos),
                                    baseline: ExpertIOStats()).isBlocking)
        }

        // ~12s blocked out of ~245s: a majority gate withholds the advisory.
        #expect(tracker.summary == nil)
    }

    @Test func aHealthyPrefillIsNotReportedAsIOBound() {
        // Measured on this machine: 12,013 tokens in 246s, 82.6s of expert
        // reads, 100 chunks. That is a ratio near 0.13, well under the line.
        let clock = FakeClock()
        let tracker = PrefillIOTracker(start: clock.now, now: { clock.now })
        for _ in 0..<3 {
            clock.advance(by: .seconds(5))
            tracker.record(cumulative: ExpertIOStats(bytesRead: 1, readNanos: 500_000_000),
                           baseline: ExpertIOStats())
        }

        #expect(tracker.summary == nil)
    }

    @Test func aSecondRequestsFirstChunkIsMeasuredAgainstItsOwnBaseline() throws {
        // The counters are cumulative for the model, so a second request starts
        // with a large non-zero total. Subtracting nothing would report its
        // first chunk as having read zero bytes over a real wall time, which
        // reads as perfectly healthy — the opposite of a cold pool.
        let clock = FakeClock()
        let carried = ExpertIOStats(bytesRead: 43 * 1_073_741_824, readNanos: 71_000_000_000,
                                    missCount: 13_875, hitCount: 35_489)
        let tracker = PrefillIOTracker(start: clock.now, now: { clock.now })
        clock.advance(by: .seconds(138))
        let chunk = tracker.record(
            cumulative: ExpertIOStats(bytesRead: carried.bytesRead + 12_884_901_888,
                                      readNanos: carried.readNanos + 469_000_000_000,
                                      missCount: carried.missCount + 128,
                                      hitCount: carried.hitCount),
            baseline: carried)

        #expect(chunk.bytes == 12_884_901_888)
        #expect(chunk.readNanos == 469_000_000_000)
        #expect(chunk.misses == 128)
        #expect(chunk.isBlocking)
        #expect(try #require(tracker.summary).blockedChunks == 1)
    }

    @Test func aChunkWhoseReadsOutlastItIsReportedAsIOBound() throws {
        // The 2 tok/s collapse: 138s of wall time per chunk carrying 469s of
        // thread-summed pread time, which is what a fully evicted 12 GB expert
        // pool looks like.
        let clock = FakeClock()
        let tracker = PrefillIOTracker(start: clock.now, now: { clock.now })
        clock.advance(by: .seconds(138))
        let blocked = tracker.record(cumulative: ExpertIOStats(bytesRead: 12_884_901_888,
                                                               readNanos: 469_000_000_000,
                                                               missCount: 128, hitCount: 0),
                                     baseline: ExpertIOStats())
        clock.advance(by: .seconds(6))
        let healthy = tracker.record(cumulative: ExpertIOStats(bytesRead: 12_885_000_000,
                                                              readNanos: 470_000_000_000,
                                                              missCount: 128, hitCount: 380),
                                     baseline: ExpertIOStats())

        #expect(blocked.isBlocking)
        #expect(!healthy.isBlocking)
        let summary = try #require(tracker.summary)
        #expect(summary.chunks == 2)
        #expect(summary.blockedChunks == 1)
        #expect(abs(summary.worstRatio - 3.4) < 0.05)
        // 138 of 144 seconds: reads owned the prefill, so the advisory applies.
        #expect(abs(summary.blockedShare - 138.0 / 144.0) < 1e-9)
    }

    @Test func aPrefillThatReportedNoChunksDoesNotClaimToBeIOBound() {
        // A request that dies before its first chunk must not produce a
        // summary: there is no evidence of I/O, and blaming the disk for a
        // failure that never reached the experts would send an operator the
        // wrong way.
        let tracker = PrefillIOTracker(start: .now, now: { .now })
        #expect(tracker.summary == nil)
    }

    @Test func theIOBoundLineNamesTheKnobsThatChangeReadVolume() {
        let line = ServerLog.prefillIOBoundMessage(
            id: "chatcmpl-abc",
            summary: PrefillIOSummary(chunks: 100, blockedChunks: 37, worstRatio: 3.42,
                                      bytesRead: 43 * 1_073_741_824,
                                      blockedWallNanos: 4_177_000_000_000,
                                      totalWallNanos: 5_458_000_000_000))

        #expect(line.contains("chatcmpl-abc"))
        #expect(line.contains("io-bound"))
        #expect(line.contains("blocked=77%"))
        #expect(line.contains("blocked_chunks=37/100"))
        #expect(line.contains("worst_ratio=3.42"))
        #expect(line.contains("expert_gb=43.00"))
        // The whole point of the line: what to change, not just what went wrong.
        #expect(line.contains("--expert-cache-slots"))
        #expect(!line.contains("failed"))
    }
}
