//
//  AsyncLayerExecutor.swift
//  TurboFieldfare
//
//  Drives the transformer decoder stack one layer at a time on a single Metal
//  command queue.
//
//  Decoder layers have a strict data dependency: layer N consumes the hidden
//  state layer N-1 produced, so layers cannot run concurrently. The pipeline
//  already overlaps routed-expert I/O with the shared-expert GPU work inside
//  `RealForwardRunner`; this type does not add a second overlap. What it adds
//  is an explicit, validated layer walk that threads the previous layer's
//  pending command buffers through as its carried state, so nothing is dropped
//  between layers, plus per-layer timing and failure attribution for
//  diagnostics.
//

import Metal
import Foundation

final class AsyncLayerExecutor: @unchecked Sendable {

    enum LayerState: Equatable {
        case pending
        case active
        case completed
        case failed(any Error)

        static func == (lhs: LayerState, rhs: LayerState) -> Bool {
            switch (lhs, rhs) {
            case (.pending, .pending), (.active, .active), (.completed, .completed):
                return true
            case (.failed, .failed):
                return true
            default:
                return false
            }
        }
    }

    /// Wall-clock cost of one layer, measured around the body call. The
    /// encode/IO split the routed-expert path knows about internally is not
    /// visible here, so this reports total layer time only.
    struct LayerStats: Equatable {
        let layer: Int
        let nanoseconds: UInt64
    }

    private let queue: MTLCommandQueue
    private let numLayers: Int

    private let lock = NSLock()
    private var _states: [Int: LayerState] = [:]
    private var _stats: [LayerStats] = []

    init(queue: MTLCommandQueue, numLayers: Int) {
        self.queue = queue
        self.numLayers = numLayers
    }

    func validate(_ layer: Int) throws {
        guard 0..<numLayers ~= layer else {
            throw ModelError.invalidLayerIndex(layer)
        }
    }

    /// Runs `body` for every layer in `layers`, threading the returned state
    /// from one layer into the next. Rejects the whole range up front so a
    /// half-applied range cannot leave later layers marked active.
    func runLayers<State>(
        _ layers: Range<Int>,
        initial state: State,
        body: (Int, State) async throws -> State
    ) async throws -> State {
        for layer in layers {
            try validate(layer)
        }

        var carried = state
        for layer in layers {
            setState(.active, for: layer)
            let start = DispatchTime.now().uptimeNanoseconds
            do {
                carried = try await body(layer, carried)
            } catch {
                let elapsed = DispatchTime.now().uptimeNanoseconds &- start
                append(LayerStats(layer: layer, nanoseconds: elapsed))
                setState(.failed(error), for: layer)
                throw error
            }
            let elapsed = DispatchTime.now().uptimeNanoseconds &- start
            append(LayerStats(layer: layer, nanoseconds: elapsed))
            setState(.completed, for: layer)
        }
        return carried
    }

    func state(_ layer: Int) -> LayerState {
        lock.lock()
        defer { lock.unlock() }
        return _states[layer] ?? .pending
    }

    var stats: [LayerStats] {
        lock.lock()
        defer { lock.unlock() }
        return _stats
    }

    func reset() {
        lock.lock()
        defer { lock.unlock() }
        _states.removeAll()
        _stats.removeAll()
    }

    private func setState(_ newValue: LayerState, for layer: Int) {
        lock.lock()
        defer { lock.unlock() }
        _states[layer] = newValue
    }

    private func append(_ stat: LayerStats) {
        lock.lock()
        defer { lock.unlock() }
        _stats.append(stat)
    }
}
