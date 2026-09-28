//
//  AsyncLayerExecutorTests.swift
//  TurboFieldfare
//
//  Tests for AsyncLayerExecutor, which drives a strictly sequential
//  decoder stack while preserving the single-queue commit order the
//  kernels rely on.
//

import XCTest
@testable import TurboFieldfare
import Metal

final class AsyncLayerExecutorTests: XCTestCase {

    var executor: AsyncLayerExecutor!
    var queue: MTLCommandQueue!

    override func setUp() {
        super.setUp()
        queue = MTLCreateSystemDefaultDevice()!.makeCommandQueue()!
        executor = AsyncLayerExecutor(queue: queue, numLayers: 4)
    }

    override func tearDown() {
        executor = nil
        queue = nil
        super.tearDown()
    }

    // MARK: - Validation

    func testValidateAcceptsInRange() throws {
        for layer in 0..<4 {
            XCTAssertNoThrow(try executor.validate(layer))
        }
    }

    func testValidateRejectsOutOfRange() {
        XCTAssertThrowsError(try executor.validate(4)) { error in
            XCTAssertEqual(error as? ModelError, .invalidLayerIndex(4))
        }
        XCTAssertThrowsError(try executor.validate(999)) { error in
            XCTAssertEqual(error as? ModelError, .invalidLayerIndex(999))
        }
    }

    // MARK: - Layer execution

    func testRunLayersReturnsAccumulatedState() async throws {
        let result = try await executor.runLayers(0..<4, initial: 0) { _, acc in
            acc + 1
        }
        XCTAssertEqual(result, 4)
    }

    func testRunLayersTransitionsToCompleted() async throws {
        XCTAssertEqual(executor.state(0), .pending)
        _ = try await executor.runLayers(0..<1, initial: ()) { _, _ in }
        XCTAssertEqual(executor.state(0), .completed)
    }

    func testRunLayersPropagatesError() async throws {
        do {
            _ = try await executor.runLayers(0..<4, initial: 0) { _, _ in
                throw PrefillError.prefillError("boom")
            }
            XCTFail("expected an error")
        } catch {
            XCTAssertEqual(error as? PrefillError, .prefillError("boom"))
        }
    }

    func testRunLayersStopsAtFirstError() async throws {
        var layersSeen = 0
        do {
            _ = try await executor.runLayers(0..<4, initial: 0) { layer, _ in
                layersSeen += 1
                if layer == 1 { throw PrefillError.prefillError("stop") }
                return layer
            }
            XCTFail("expected an error")
        } catch {
            XCTAssertEqual(error as? PrefillError, .prefillError("stop"))
        }
        XCTAssertEqual(layersSeen, 2)
    }

    // MARK: - State tracking

    func testInitialStateIsPending() {
        XCTAssertEqual(executor.state(0), .pending)
    }

    func testStateAfterRun() async throws {
        _ = try await executor.runLayers(0..<3, initial: ()) { _, _ in }
        for layer in 0..<3 {
            XCTAssertEqual(executor.state(layer), .completed)
        }
    }

    // MARK: - Reset

    func testResetClearsState() async throws {
        _ = try await executor.runLayers(0..<2, initial: ()) { _, _ in }
        executor.reset()
        for layer in 0..<2 {
            XCTAssertEqual(executor.state(layer), .pending)
        }
    }

    // MARK: - Layer stats

    func testLayerStatsRecordedPerLayer() async throws {
        _ = try await executor.runLayers(0..<3, initial: ()) { _, _ in }
        let stats = executor.stats
        XCTAssertEqual(stats.count, 3)
        XCTAssertEqual(stats.map(\.layer), [0, 1, 2])
    }

    func testFailedLayerIsRecordedAndStopsTheWalk() async throws {
        do {
            _ = try await executor.runLayers(0..<4, initial: ()) { layer, _ in
                if layer == 1 { throw PrefillError.prefillError("stop") }
            }
            XCTFail("expected an error")
        } catch {
            XCTAssertEqual(error as? PrefillError, .prefillError("stop"))
        }
        XCTAssertEqual(executor.state(0), .completed)
        XCTAssertEqual(executor.state(1), .failed(PrefillError.prefillError("stop")))
        XCTAssertEqual(executor.state(2), .pending)
        XCTAssertEqual(executor.state(3), .pending)
        XCTAssertEqual(executor.stats.map(\.layer), [0, 1])
    }

    func testOutOfRangeLayerIsRejectedBeforeAnyLayerRuns() async throws {
        do {
            _ = try await executor.runLayers(4..<6, initial: ()) { _, _ in }
            XCTFail("expected an error")
        } catch {
            XCTAssertEqual(error as? ModelError, .invalidLayerIndex(4))
        }
        XCTAssertEqual(executor.stats.count, 0)
    }

    func testResetClearsStats() async throws {
        _ = try await executor.runLayers(0..<2, initial: ()) { _, _ in }
        XCTAssertEqual(executor.stats.count, 2)
        executor.reset()
        XCTAssertEqual(executor.stats.count, 0)
    }
}
