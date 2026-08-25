import Foundation
import Metal

public final class Qwen35ForwardRunner: ChunkedPrefillRunner,
                                          ContextWindowReporting,
                                          ContinuableLogitProducer,
                                          @unchecked Sendable {
    private let model: Model
    private let ctx: MetalContext
    private let kv: KVCacheManager?
    private let cfg: ArchConfig

    private let embedInt4: EmbedLookupInt4
    private let rms: RMSNorm
    private let int4: DequantInt4GEMV
    private let attention: Attention
    private let shared: SharedExpertRuntime
    private let moe: MoE
    private let fusionHead: LMHeadChainInt4

    private let qwenAttnQK: QwenAttnQKEpilogue
    private let qwenGateMul: QwenSigmoidGateMul
    private let qwenRouter: QwenRouterSelect
    private let qwenGDN: QwenGDNForward
    private let qwenResidualAdd: QwenResidualAddBatch
    private let qwenFFNCombine: QwenFFNCombineBatch

    private let prefillEmbed: PrefillEmbedLookupInt4
    private let prefillRMS: PrefillRMSNorm
    private let prefillQMM: PrefillInt4QMM
    private let prefillAttention: PrefillAttention
    private let prefillFinalRowHead: PrefillFinalRowHeadInt4
    private let prefillSharedExpert: PrefillSharedExpert
    private let prefillGroupedMoE: PrefillGroupedRoutedMoE
    private let prefillMoE: PrefillMoE
    private let prefillLayerTail: PrefillLayerTail

    private let hidden: MTLBuffer
    private let normed: MTLBuffer
    private let qScratch: MTLBuffer
    private let qCompact: MTLBuffer
    private let kStage: MTLBuffer
    private let vStage: MTLBuffer
    private let attnOut: MTLBuffer
    private let oOut: MTLBuffer
    private let h1Buf: MTLBuffer
    private let gdnProjBuf: MTLBuffer
    private let h2Buf: MTLBuffer
    private let denseX: MTLBuffer
    private let routedX: MTLBuffer
    private let routerInput: MTLBuffer
    private let zeroResidual: MTLBuffer
    private let outIndices: MTLBuffer
    private let outWeights: MTLBuffer
    private let moeActs: MTLBuffer
    private let moeHitActiveSlots: MTLBuffer
    private let moeMissActiveSlots: MTLBuffer
    private let greedyTokenBuf: MTLBuffer
    private let denseScratchGate: MTLBuffer
    private let denseScratchUp: MTLBuffer
    private let denseScratchAct: MTLBuffer
    private let gdnConvState: [MTLBuffer]
    private let gdnState: [MTLBuffer]
    private let dumpStaging: MTLBuffer

    private struct LayerProjections {
        let gate: SharedExpertInt8Proj
        let up: SharedExpertInt8Proj
        let down: SharedExpertInt8Proj
        let gateVec: TensorView
    }
    private let sharedExpertProjections: [LayerProjections]
    // Qwen3.5's shared-expert combine gate (mlp.shared_expert_gate.weight) is
    // stored int4 like every other weight, but qwen_ffn_combine_batch reads the
    // gate as FP16. Gemma stores this gate already in FP16. We dequantize the
    // int4 gate to an FP16 buffer once at init so the combine gets valid data.
    private let gateVecHalf: [MTLBuffer]

    private var prefillChunkState = PrefillChunkCommitState()
    private var prefillScratch: PrefillChunkScratchBuffers?
    private static let prefillRoutedTileSchedulerConfig = PrefillRoutedTileSchedulerConfig()

    public let maxContext: Int
    private let useFusedGreedyHead: Bool
    public private(set) var lastGreedyToken: UInt32 = 0
    public var usesFusedGreedyHead: Bool { useFusedGreedyHead }
    private var continuationPos: Int = 0
    public var continuationPosition: Int { continuationPos }

    public init(model: Model, context: MetalContext, maxContext: Int,
                runtimeConfiguration: RuntimeConfiguration = .production) throws {
        self.model = model
        self.ctx = context
        self.cfg = model.config
        self.maxContext = maxContext
        self.useFusedGreedyHead = runtimeConfiguration.headPath == .fusedRows
        self.kv = try KVCacheManager(device: context.device,
                                     config: cfg, maxContext: maxContext,
                                     fp16RingEnabled: runtimeConfiguration.fp16RingEnabled,
                                     slidingWindow: cfg.slidingWindow,
                                     maxPrefillChunkTokens: PrefillRuntimeConfig.maxChunkTokens)

        self.embedInt4 = try EmbedLookupInt4(context: context)
        self.rms = try RMSNorm(context: context)
        self.int4 = try DequantInt4GEMV(context: context)
        self.attention = try Attention(context: context)
        self.shared = try SharedExpertRuntime(context: context,
                                                weightBits: model.sharedExpertWeightBits)
        self.moe = try MoE(context: context)
        self.fusionHead = try LMHeadChainInt4(context: context)
        self.qwenAttnQK = try QwenAttnQKEpilogue(context: context)
        self.qwenGateMul = try QwenSigmoidGateMul(context: context)
        self.qwenRouter = try QwenRouterSelect(context: context)
        self.qwenGDN = try QwenGDNForward(context: context)
        self.qwenResidualAdd = try QwenResidualAddBatch(context: context)
        self.qwenFFNCombine = try QwenFFNCombineBatch(context: context)
        self.prefillEmbed = try PrefillEmbedLookupInt4(context: context)
        self.prefillRMS = try PrefillRMSNorm(context: context)
        self.prefillQMM = try PrefillInt4QMM(context: context)
        self.prefillAttention = try PrefillAttention(context: context)
        self.prefillFinalRowHead = try PrefillFinalRowHeadInt4(context: context, maxD: cfg.hiddenSize)
        self.prefillSharedExpert = try PrefillSharedExpert(context: context,
                                                           weightBits: model.sharedExpertWeightBits)
        self.prefillGroupedMoE = try PrefillGroupedRoutedMoE(context: context)
        self.prefillMoE = try PrefillMoE(context: context)
        self.prefillLayerTail = try PrefillLayerTail(context: context)

        let D = cfg.hiddenSize
        let qRawElems = cfg.numHeads * cfg.headDim * 2
        let qCompactElems = cfg.numHeads * cfg.headDim
        let kvElems = cfg.numKVHeads * cfg.headDim
        let moElems = cfg.topKExperts * cfg.moeIntermediateSize

        func db(_ elems: Int, _ label: String) throws -> MTLBuffer {
            guard let b = context.device.makeBuffer(length: elems * MemoryLayout<Float16>.stride,
                                                    options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            b.label = label; return b
        }
        func dbRaw(_ bytes: Int, _ label: String) throws -> MTLBuffer {
            guard let b = context.device.makeBuffer(length: bytes, options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            b.label = label; return b
        }

        self.hidden = try db(D, "qwen.hidden")
        self.normed = try db(D, "qwen.normed")
        self.qScratch = try db(qRawElems, "qwen.qRaw")
        self.qCompact = try db(qCompactElems, "qwen.qCompact")
        self.kStage = try db(kvElems, "qwen.kStage")
        self.vStage = try db(kvElems, "qwen.vStage")
        self.attnOut = try db(qCompactElems, "qwen.attnOut")
        self.oOut = try db(D, "qwen.oOut")
        let gdnInternal = cfg.linearNumValueHeads * cfg.linearHeadDim
        self.h1Buf = try db(max(D, gdnInternal), "qwen.h1")
        self.gdnProjBuf = try db(D, "qwen.gdnProj")
        self.h2Buf = try db(D, "qwen.h2")
        self.denseX = try db(D, "qwen.denseX")
        self.routedX = try db(D, "qwen.routedX")
        self.routerInput = try db(D, "qwen.routerX")
        self.zeroResidual = try db(D, "qwen.zeroResid")
        self.outIndices = try dbRaw(cfg.topKExperts * MemoryLayout<UInt32>.stride, "qwen.outIdx")
        self.outWeights = try db(cfg.topKExperts, "qwen.outW")
        self.moeActs = try db(moElems, "qwen.moeActs")
        self.moeHitActiveSlots = try dbRaw(cfg.topKExperts * MemoryLayout<UInt32>.stride, "qwen.hitSlots")
        self.moeMissActiveSlots = try dbRaw(cfg.topKExperts * MemoryLayout<UInt32>.stride, "qwen.missSlots")
        self.greedyTokenBuf = try dbRaw(MemoryLayout<UInt32>.stride, "qwen.greedy")
        self.denseScratchGate = try db(cfg.intermediateSize, "qwen.dGate")
        self.denseScratchUp = try db(cfg.intermediateSize, "qwen.dUp")
        self.denseScratchAct = try db(cfg.intermediateSize, "qwen.dAct")

        let convDim = (cfg.linearNumKeyHeads * cfg.linearHeadDim * 2
                       + cfg.linearNumValueHeads * cfg.linearHeadDim)
        let kernelDim = cfg.linearConvKernelDim
        var gdnConvStates: [MTLBuffer] = []
        var gdnStates: [MTLBuffer] = []
        gdnConvStates.reserveCapacity(cfg.numLayers)
        gdnStates.reserveCapacity(cfg.numLayers)
        for l in 0..<cfg.numLayers {
            gdnConvStates.append(try dbRaw((kernelDim - 1) * convDim * MemoryLayout<Float16>.stride,
                                           "qwen.gdnConvState.\(l)"))
            gdnStates.append(try dbRaw(cfg.linearNumValueHeads * cfg.linearHeadDim * cfg.linearHeadDim
                                        * MemoryLayout<Float>.stride, "qwen.gdnState.\(l)"))
        }
        self.gdnConvState = gdnConvStates
        self.gdnState = gdnStates
        self.dumpStaging = try dbRaw(D * MemoryLayout<Float16>.stride, "qwen.dumpStaging")


        var projections: [LayerProjections] = []
        projections.reserveCapacity(cfg.numLayers)
        for L in 0..<cfg.numLayers {
            let gV = try model.qwenSharedExpertGate(layer: L)
            let uV = try model.qwenSharedExpertUp(layer: L)
            let dV = try model.qwenSharedExpertDown(layer: L)
            let gvV = try model.sharedExpertGateVec(layer: L)
            projections.append(LayerProjections(
                gate: SharedExpertInt8Proj(weights: gV.buffer, scales: gV.buffer, biases: gV.buffer,
                    weightsOffset: Int(gV.offset), scalesOffset: Int(gV.scaleOffset),
                    biasesOffset: Int(gV.biasOffset),
                    rows: UInt32(cfg.moeIntermediateSize), cols: UInt32(cfg.hiddenSize)),
                up: SharedExpertInt8Proj(weights: uV.buffer, scales: uV.buffer, biases: uV.buffer,
                    weightsOffset: Int(uV.offset), scalesOffset: Int(uV.scaleOffset),
                    biasesOffset: Int(uV.biasOffset),
                    rows: UInt32(cfg.moeIntermediateSize), cols: UInt32(cfg.hiddenSize)),
                down: SharedExpertInt8Proj(weights: dV.buffer, scales: dV.buffer, biases: dV.buffer,
                    weightsOffset: Int(dV.offset), scalesOffset: Int(dV.scaleOffset),
                    biasesOffset: Int(dV.biasOffset),
                    rows: UInt32(cfg.hiddenSize), cols: UInt32(cfg.moeIntermediateSize)),
                gateVec: gvV))
        }
        self.sharedExpertProjections = projections

        var gates: [MTLBuffer] = []
        gates.reserveCapacity(cfg.numLayers)
        for L in 0..<cfg.numLayers {
            let gv = try model.sharedExpertGateVec(layer: L)
            gates.append(try Self.buildGateHalf(gv, d: D, device: ctx.device))
        }
        self.gateVecHalf = gates
    }

    private static func buildGateHalf(_ gv: TensorView, d: Int, device: MTLDevice) throws -> MTLBuffer {
        let N = d
        let groupSize = 64
        let contents = gv.buffer.contents()
        var out = [Float16](repeating: 0, count: N)
        if gv.dtype == 0 {
            let base = contents.advanced(by: Int(gv.offset)).assumingMemoryBound(to: UInt8.self)
            let sPtr = contents.advanced(by: Int(gv.scaleOffset)).assumingMemoryBound(to: UInt16.self)
            let bPtr = contents.advanced(by: Int(gv.biasOffset)).assumingMemoryBound(to: UInt16.self)
            for i in 0..<N {
                let byte = base[i / 2]
                let nib = Float32((i & 1) == 0 ? (byte & 0x0F) : (byte >> 4))
                let g = i / groupSize
                let s = Float32(bitPattern: UInt32(sPtr[g]) << 16)
                let b = Float32(bitPattern: UInt32(bPtr[g]) << 16)
                out[i] = Float16(nib * s + b)
            }
        } else if gv.dtype == 1 {
            let bf = contents.advanced(by: Int(gv.offset)).assumingMemoryBound(to: UInt16.self)
            for i in 0..<N { out[i] = Float16(bitPattern: bf[i]) }
        } else {
            let cnt = N * 2
            guard let buf = device.makeBuffer(bytes: contents.advanced(by: Int(gv.offset)),
                                                      length: cnt, options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            return buf
        }
        return try out.withUnsafeBytes { raw in
            guard let buf = device.makeBuffer(bytes: raw.baseAddress!,
                                                      length: raw.count, options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            return buf
        }
    }

    public func reset() {
        kv?.reset()
        continuationPos = 0
        prefillChunkState.reset()
        zeroBuffer(hidden)
        for b in gdnConvState { zeroBuffer(b) }
        for b in gdnState { zeroBuffer(b) }
    }

    private func zeroBuffer(_ buf: MTLBuffer) { memset(buf.contents(), 0, buf.length) }

    private func dumpRaw(_ buf: MTLBuffer, rows: Int, cols: Int, label: String) {
        guard ProcessInfo.processInfo.environment["TFDUMP"] != nil else { return }
        let total = min(rows * cols, Int(buf.length) / MemoryLayout<Float16>.stride)
        guard total > 0, let stg = ctx.device.makeBuffer(length: total * MemoryLayout<Float16>.stride,
                                                          options: .storageModeShared) else { return }
        let cb = ctx.queue.makeCommandBuffer()!
        let blit = cb.makeBlitCommandEncoder()!
        blit.copy(from: buf, sourceOffset: 0, to: stg, destinationOffset: 0, size: total * MemoryLayout<Float16>.stride)
        blit.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        let ptr = stg.contents().bindMemory(to: Float16.self, capacity: total)
        let arr = Array(UnsafeBufferPointer(start: ptr, count: total))
        let url = URL(fileURLWithPath: "/tmp/qwen_raw_\(label).bin")
        let data = arr.withUnsafeBytes { Data(bytes: $0.baseAddress!, count: $0.count) }
        try? data.write(to: url)
    }

    private func dumpStats(_ buf: MTLBuffer, rows: Int, cols: Int, label: String,
                           offsetElems: Int = 0) {
        guard ProcessInfo.processInfo.environment["TFDUMP"] != nil else { return }
        let cb = ctx.queue.makeCommandBuffer()!
        let blit = cb.makeBlitCommandEncoder()!
        blit.copy(from: buf, sourceOffset: offsetElems * MemoryLayout<Float16>.stride,
                  to: dumpStaging, destinationOffset: 0,
                  size: cols * MemoryLayout<Float16>.stride)
        blit.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        let ptr = dumpStaging.contents().bindMemory(to: Float16.self, capacity: cols)
        var mx: Float = -Float.infinity, mn: Float = Float.infinity, sum: Float = 0
        var nan = 0
        for i in 0..<cols {
            let v = Float(ptr[i])
            if v.isNaN || v.isInfinite { nan += 1 } else { mx = max(mx, v); mn = min(mn, v); sum += v }
        }
        let s = "\(label): nan=\(nan) min=\(mn) max=\(mx) mean=\(sum / Float(cols))\n"
        let url = URL(fileURLWithPath: "/tmp/qwen_dump.txt")
        if let fh = try? FileHandle(forWritingTo: url) {
            fh.seekToEndOfFile()
            fh.write(Data(s.utf8))
            try? fh.close()
        } else {
            try? s.data(using: .utf8)?.write(to: url)
        }
    }

    private func dumpStatsBF16(_ buf: MTLBuffer, cols: Int, label: String,
                               offsetElems: Int = 0) {
        guard ProcessInfo.processInfo.environment["TFDUMP"] != nil else { return }
        let cb = ctx.queue.makeCommandBuffer()!
        let blit = cb.makeBlitCommandEncoder()!
        blit.copy(from: buf, sourceOffset: offsetElems * MemoryLayout<UInt16>.stride,
                  to: dumpStaging, destinationOffset: 0,
                  size: cols * MemoryLayout<UInt16>.stride)
        blit.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        let ptr = dumpStaging.contents().bindMemory(to: UInt16.self, capacity: cols)
        var mx: Float = -Float.infinity, mn: Float = Float.infinity, sum: Float = 0
        var nan = 0
        for i in 0..<cols {
            let f = Float(bitPattern: UInt32(ptr[i]) << 16)
            if f.isNaN || f.isInfinite { nan += 1 } else { mx = max(mx, f); mn = min(mn, f); sum += f }
        }
        let s = "\(label)[bf16]: nan=\(nan) min=\(mn) max=\(mx) mean=\(sum / Float(cols))\n"
        let url = URL(fileURLWithPath: "/tmp/qwen_dump.txt")
        if let fh = try? FileHandle(forWritingTo: url) {
            fh.seekToEndOfFile()
            fh.write(Data(s.utf8))
            try? fh.close()
        } else {
            try? s.data(using: .utf8)?.write(to: url)
        }
    }

    public func produce(token: Int32, position: Int, into logits: MTLBuffer) async throws {
        try prefillChunkState.requireClean(operation: "produce")
        try await produceToken(token: token, position: position, into: logits,
                               emitHead: true, outputMode: .greedyIfAvailable)
    }

    public func prepareForContinuation(expectedPosition: Int) throws {
        let kvPos = kv?.position ?? 0
        guard kvPos == expectedPosition else {
            throw PrefillError.prefillCursorMismatch(
                "prepareForContinuation cursor \(kvPos) != \(expectedPosition)")
        }
        continuationPos = expectedPosition
    }

    public func prefillChunked(tokens: ArraySlice<Int32>,
                               startPosition: Int,
                               outputMode: PrefillOutputMode,
                               config: PrefillRuntimeConfig,
                               into logits: MTLBuffer,
                               onProgress: (Int) -> Void) async throws -> PrefillResult {
        try prefillChunkState.requireClean(operation: "prefillChunked")
        guard config.mode == .chunked else {
            throw PrefillError.chunkedUnsupported("prefillChunked requires .chunked mode")
        }
        let kvPos = kv?.position ?? 0
        guard kvPos == startPosition else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill cursor \(kvPos) != startPosition \(startPosition)")
        }
        guard tokens.count <= maxContext - startPosition else {
            throw PrefillError.chunkedUnsupported("chunked prefill range exceeds maxContext")
        }
        guard !tokens.isEmpty else {
            return PrefillResult(newPosition: startPosition, seed: .logitsWritten)
        }
        let scratch = try ensurePrefillScratch(config: config)
        let spans = PrefillChunkPlanner.spans(tokenCount: tokens.count,
                                              startPosition: startPosition, config: config)
        for (spanIdx, span) in spans.enumerated() {
            let lower = tokens.index(tokens.startIndex, offsetBy: span.tokenOffset)
            let upper = tokens.index(lower, offsetBy: span.tokenCount)
            try await executePrefillChunk(tokens: tokens[lower..<upper],
                                          startPosition: span.startPosition,
                                          outputMode: outputMode, logits: logits,
                                          scratch: scratch, config: config,
                                          writeFinalHead: spanIdx == spans.count - 1)
            onProgress(span.completedCount)
        }
        if outputMode == .greedyIfAvailable, useFusedGreedyHead {
            return PrefillResult(newPosition: startPosition + tokens.count,
                                 seed: .greedyToken(lastGreedyToken))
        }
        return PrefillResult(newPosition: startPosition + tokens.count, seed: .logitsWritten)
    }

    @discardableResult
    private func ensurePrefillScratch(config: PrefillRuntimeConfig) throws -> PrefillChunkScratchBuffers {
        let layout = PrefillChunkScratchLayout(config: cfg, runtime: config)
        if let s = prefillScratch, s.layout == layout { return s }
        let s = try PrefillChunkScratchBuffers.allocate(device: ctx.device, layout: layout)
        prefillScratch = s; return s
    }

    private func runSync(_ body: (MTLCommandBuffer) -> Void) throws {
        let cb = ctx.queue.makeCommandBuffer()!
        body(cb); cb.commit(); cb.waitUntilCompleted()
        try checkCommandBufferError(cb.error)
    }

    private func wait(_ cb: MTLCommandBuffer) throws {
        cb.waitUntilCompleted(); try checkCommandBufferError(cb.error)
    }

    private func produceToken(token: Int32, position: Int, into logits: MTLBuffer,
                              emitHead: Bool, outputMode: PrefillOutputMode) async throws {
        let D = UInt32(cfg.hiddenSize)
        let eps: Float = 1e-6
        let sqrtHidden = Float(D).squareRoot()
        let emb = model.embedding

        try runSync { cb in
            embedInt4.encode(commandBuffer: cb,
                             table: emb.buffer, tableOffset: Int(emb.offset),
                             scales: emb.buffer, scalesOffset: Int(emb.scaleOffset),
                             biases: emb.buffer, biasesOffset: Int(emb.biasOffset),
                             out: hidden, tokenId: UInt32(bitPattern: token),
                             d: D, outScale: sqrtHidden)
        }

        let isFullAttnLayer: (Int) -> Bool = { L in self.cfg.fullAttentionLayerMask[L] != 0 }

        for L in 0..<cfg.numLayers {
            let isFull = isFullAttnLayer(L)
            if isFull {
                try await runFullAttentionLayer(L, position: position, D: D, eps: eps)
            } else {
                try await runGDNLayer(L, position: position, D: D, eps: eps)
            }
        }

        if emitHead {
            let fn = model.finalNorm
            let lm = model.lmHead
            let useFused = useFusedGreedyHead && outputMode == .greedyIfAvailable
            try runSync { cb in
                if useFused {
                    fusionHead.encodeGreedyDecode(
                        commandBuffer: cb, hidden: hidden,
                        normWeight: fn.buffer, normOffset: Int(fn.offset),
                        weights: lm.buffer, weightsOffset: Int(lm.offset),
                        scales: lm.buffer, scalesOffset: Int(lm.scaleOffset),
                        biases: lm.buffer, biasesOffset: Int(lm.biasOffset),
                        outToken: greedyTokenBuf,
                        d: D, vocab: UInt32(cfg.vocabSize), rmsEps: eps)
                } else {
                    rms.encodeBF16W(commandBuffer: cb, x: hidden,
                                    weight: fn.buffer, weightOffset: Int(fn.offset),
                                    out: normed, d: D, eps: eps)
                    int4.encode(commandBuffer: cb,
                                weights: lm.buffer, weightsOffset: Int(lm.offset),
                                scales: lm.buffer, scalesOffset: Int(lm.scaleOffset),
                                biases: lm.buffer, biasesOffset: Int(lm.biasOffset),
                                x: normed, y: logits,
                                m: UInt32(cfg.vocabSize), n: D)
                }
            }
            if useFused {
                lastGreedyToken = greedyTokenBuf.contents().load(as: UInt32.self)
            }
        }
        kv?.advance()
    }

    private func runFullAttentionLayer(_ L: Int, position: Int,
                                       D: UInt32, eps: Float) async throws {
        let headDimL = cfg.headDim
        let numKVL = cfg.numKVHeads
        let qDim = UInt32(cfg.numHeads * headDimL)
        let kvDim = UInt32(numKVL * headDimL)
        let seqLen = UInt32(position + 1)
        let inNorm = try model.inputNorm(layer: L)
        let qW = try model.qProj(layer: L)
        let kW = try model.kProj(layer: L)
        let vW = try model.vProj(layer: L)
        let oW = try model.oProj(layer: L)
        let postAttn = try model.postAttnNorm(layer: L)
        let qNormV = try model.qNorm(layer: L)
        let kNormV = try model.kNorm(layer: L)
        let proj = sharedExpertProjections[L]
        
        let kSlot = kv?.kSlot(layer: L, position: position) ?? (buffer: kStage, offset: 0)
        let vSlot = kv?.vSlot(layer: L, position: position) ?? (buffer: vStage, offset: 0)
        let rotatedPairs = UInt32(Double(headDimL) * cfg.partialRotaryFactor / 2.0)

        try runSync { cb in
            // input norm
            rms.encodeBF16W(commandBuffer: cb, x: hidden,
                            weight: inNorm.buffer, weightOffset: Int(inNorm.offset),
                            out: normed, d: D, eps: eps)
            // q_proj (raw, includes gate halves)
            int4.encode(commandBuffer: cb,
                        weights: qW.buffer, weightsOffset: Int(qW.offset),
                        scales: qW.buffer, scalesOffset: Int(qW.scaleOffset),
                        biases: qW.buffer, biasesOffset: Int(qW.biasOffset),
                        x: normed, y: qScratch, m: qDim * 2, n: D)
            // k_proj
            int4.encode(commandBuffer: cb,
                        weights: kW.buffer, weightsOffset: Int(kW.offset),
                        scales: kW.buffer, scalesOffset: Int(kW.scaleOffset),
                        biases: kW.buffer, biasesOffset: Int(kW.biasOffset),
                        x: normed, y: kStage, yOffset: 0,
                        m: kvDim, n: D)
            // v_proj
            int4.encode(commandBuffer: cb,
                        weights: vW.buffer, weightsOffset: Int(vW.offset),
                        scales: vW.buffer, scalesOffset: Int(vW.scaleOffset),
                        biases: vW.buffer, biasesOffset: Int(vW.biasOffset),
                        x: normed, y: vSlot.buffer, yOffset: vSlot.offset,
                        m: kvDim, n: D)

            // q/k epilogue: per-head RMSNorm + partial-segment RoPE
            qwenAttnQK.encode(
                commandBuffer: cb,
                qRaw: qScratch, kRaw: kStage,
                qWeight: qNormV.buffer, qWeightOffset: Int(qNormV.offset),
                kWeight: kNormV.buffer, kWeightOffset: Int(kNormV.offset),
                qOut: qCompact, kOut: kSlot.buffer, kOutOffset: kSlot.offset,
                headDim: UInt32(headDimL), numQHeads: UInt32(cfg.numHeads),
                numKVHeads: UInt32(numKVL), rotatedPairs: rotatedPairs,
                theta: Float(cfg.ropeTheta), positionBase: UInt32(position),
                tokenCount: 1, eps: eps)

            // attention
            let ringCap = kv?.ringCapacity(layer: L) ?? 0
            let activeRing = ringCap > 0 && Int(seqLen) > ringCap ? UInt32(ringCap) : 0
            attention.encodeSWA(commandBuffer: cb,
                                q: qCompact,
                                k: kSlot.buffer, kOffset: 0,
                                v: vSlot.buffer, vOffset: 0,
                                out: attnOut,
                                headDim: UInt32(headDimL),
                                numQHeads: UInt32(cfg.numHeads),
                                numKVHeads: UInt32(numKVL),
                                seqLen: seqLen,
                                window: UInt32(cfg.slidingWindow),
                                scale: 1.0 / sqrt(Float(headDimL)), ringCapacity: activeRing)

            // sigmoid gate multiply: attn *= sigmoid(gate half of qRaw)
            qwenGateMul.encode(commandBuffer: cb,
                               attn: attnOut, qRaw: qScratch,
                               headDim: UInt32(headDimL),
                               numQHeads: UInt32(cfg.numHeads),
                               tokenCount: 1)

            // o_proj
            int4.encode(commandBuffer: cb,
                        weights: oW.buffer, weightsOffset: Int(oW.offset),
                        scales: oW.buffer, scalesOffset: Int(oW.scaleOffset),
                        biases: oW.buffer, biasesOffset: Int(oW.biasOffset),
                        x: attnOut, y: oOut, m: D, n: qDim)

            // attention residual: hidden = hidden + oOut
            qwenResidualAdd.encode(commandBuffer: cb,
                                   hidden: hidden, delta: oOut,
                                   count: Int(D))

            // post-attention norm (single Qwen3.5 norm) for all FFN inputs
            rms.encodeBF16W(commandBuffer: cb, x: hidden,
                            weight: postAttn.buffer, weightOffset: Int(postAttn.offset),
                            out: denseX, d: D, eps: eps)
            rms.encodeBF16W(commandBuffer: cb, x: hidden,
                            weight: postAttn.buffer, weightOffset: Int(postAttn.offset),
                            out: routedX, d: D, eps: eps)
            rms.encodeBF16W(commandBuffer: cb, x: hidden,
                            weight: postAttn.buffer, weightOffset: Int(postAttn.offset),
                            out: routerInput, d: D, eps: eps)
        }

        // Router readback
        try runSync { cb in
            let inputScale = 1.0 / sqrt(Float(cfg.hiddenSize))
            qwenRouter.encode(commandBuffer: cb,
                              logits: routerInput,
                              outIndices: outIndices, outWeights: outWeights,
                              numExperts: UInt32(cfg.numExperts), topK: UInt32(cfg.topKExperts),
                              inputScale: inputScale, tokenCount: 1)
        }

        let idxPtr = outIndices.contents().bindMemory(to: UInt32.self, capacity: cfg.topKExperts)
        var experts = [Int](repeating: 0, count: cfg.topKExperts)
        for i in 0..<cfg.topKExperts { experts[i] = min(Int(idxPtr[i]), cfg.numExperts - 1) }

        let routedOffsets = model.routedExpertOffsets(layer: L)
        let topK = UInt32(cfg.topKExperts)

        // Shared expert (parallel with routed I/O)
        let sharedCB = ctx.queue.makeCommandBuffer()!
        try shared.encode(commandBuffer: sharedCB,
                          x: denseX, gate: proj.gate, up: proj.up, down: proj.down,
                          y: h1Buf,
                          scratchGate: denseScratchGate, scratchUp: denseScratchUp,
                          scratchAct: denseScratchAct)
        sharedCB.commit()

        // Routed expert fetch
        let blobs = try await model.fetchRoutedExperts(layer: L, experts: experts)
        let routedBufs = blobs.map { $0.buffer }

        try runSync { cb in
            let argBuf = moe.makeReusedRoutedArgumentBuffer(routedBlobs: routedBufs, topK: topK)
            moe.encodeRoutedPersistentPhase1U16Load(
                commandBuffer: cb, routedArgBuffer: argBuf, routedBlobs: routedBufs,
                routedOffsets: routedOffsets, x: routedX, acts: moeActs,
                d: D, f: UInt32(cfg.moeIntermediateSize), topK: topK)
            moe.encodeRoutedPersistentPhase2Reduce(
                commandBuffer: cb, routedArgBuffer: argBuf, routedBlobs: routedBufs,
                routedOffsets: routedOffsets, acts: moeActs, routingWeights: outWeights,
                residual: zeroResidual, y: h2Buf,
                d: D, f: UInt32(cfg.moeIntermediateSize), topK: topK)

            // combine: hidden += h2 + h1 * sigmoid(gate_vec)
            qwenFFNCombine.encode(commandBuffer: cb,
                                  hidden: hidden, routed: h2Buf, shared: h1Buf,
                                  gate: self.gateVecHalf[L],
                                  d: Int(D), count: Int(D))
        }
    }

    private func runGDNLayer(_ L: Int, position: Int,
                             D: UInt32, eps: Float) async throws {
        let inNorm = try model.inputNorm(layer: L)
        let inQKV = try model.inProjQKV(layer: L)
        let inZ = try model.inProjZ(layer: L)
        let inA = try model.inProjA(layer: L)
        let inB = try model.inProjB(layer: L)
        let convW = try model.gdnConvWeight(layer: L)
        let aLogV = try model.gdnALog(layer: L)
        let dtBiasV = try model.gdnDtBias(layer: L)
        let normW = try model.gdnNormWeight(layer: L)
        let postAttn = try model.postAttnNorm(layer: L)
        let proj = sharedExpertProjections[L]
        
        let outProj = try model.qwenGDNOutProj(layer: L)

        let convDim = UInt32(cfg.linearNumKeyHeads * cfg.linearHeadDim * 2
                             + cfg.linearNumValueHeads * cfg.linearHeadDim)

        // Each GDN layer keeps its own recurrent state in gdnState[L] /
        // gdnConvState[L], which persists across tokens (reset only on reset()).

        // Project: mixed = in_proj_qkv(normed), z = in_proj_z(normed),
        //          a = in_proj_a(normed), b = in_proj_b(normed)
        let mixed = normed
        let zBuf = h2Buf
        let aBuf = outWeights
        let bBuf = outIndices

        try runSync { cb in
            rms.encodeBF16W(commandBuffer: cb, x: hidden,
                            weight: inNorm.buffer, weightOffset: Int(inNorm.offset),
                            out: normed, d: D, eps: eps)
            int4.encode(commandBuffer: cb,
                        weights: inQKV.buffer, weightsOffset: Int(inQKV.offset),
                        scales: inQKV.buffer, scalesOffset: Int(inQKV.scaleOffset),
                        biases: inQKV.buffer, biasesOffset: Int(inQKV.biasOffset),
                        x: normed, y: mixed,
                        m: convDim, n: D)
            int4.encode(commandBuffer: cb,
                        weights: inZ.buffer, weightsOffset: Int(inZ.offset),
                        scales: inZ.buffer, scalesOffset: Int(inZ.scaleOffset),
                        biases: inZ.buffer, biasesOffset: Int(inZ.biasOffset),
                        x: normed, y: zBuf,
                        m: UInt32(cfg.linearNumValueHeads * cfg.linearHeadDim), n: D)
            int4.encode(commandBuffer: cb,
                        weights: inA.buffer, weightsOffset: Int(inA.offset),
                        scales: inA.buffer, scalesOffset: Int(inA.scaleOffset),
                        biases: inA.buffer, biasesOffset: Int(inA.biasOffset),
                        x: normed, y: aBuf,
                        m: UInt32(cfg.linearNumValueHeads), n: D)
            int4.encode(commandBuffer: cb,
                        weights: inB.buffer, weightsOffset: Int(inB.offset),
                        scales: inB.buffer, scalesOffset: Int(inB.scaleOffset),
                        biases: inB.buffer, biasesOffset: Int(inB.biasOffset),
                        x: normed, y: bBuf,
                        m: UInt32(cfg.linearNumValueHeads), n: D)
        }

        let inputScale = 1.0 / sqrt(Float(cfg.linearHeadDim))
        try runSync { cb in
            qwenGDN.encode(commandBuffer: cb,
                           mixed: mixed, z: zBuf,
                           aVec: aBuf, bVec: bBuf,
                           convWeight: convW.buffer, convWeightOffset: Int(convW.offset),
                            convState: gdnConvState[L], state: gdnState[L],
                           aLog: aLogV.buffer, aLogOffset: Int(aLogV.offset),
                           dtBias: dtBiasV.buffer, dtBiasOffset: Int(dtBiasV.offset),
                           normWeight: normW.buffer, normWeightOffset: Int(normW.offset),
                           out: h1Buf,
                           tokenCount: 1,
                           valueHeads: UInt32(cfg.linearNumValueHeads),
                           keyHeads: UInt32(cfg.linearNumKeyHeads),
                           convDim: convDim,
                           kernelDim: UInt32(cfg.linearConvKernelDim),
                           eps: eps, inputScale: inputScale)
        }

        // GDN out_proj: gated per-head output (num_value_heads*value_head_dim)
        // -> hidden, then residual add.
        let gdnInternal = cfg.linearNumValueHeads * cfg.linearHeadDim
        try runSync { cb in
            int4.encode(commandBuffer: cb,
                        weights: outProj.buffer, weightsOffset: Int(outProj.offset),
                        scales: outProj.buffer, scalesOffset: Int(outProj.scaleOffset),
                        biases: outProj.buffer, biasesOffset: Int(outProj.biasOffset),
                        x: h1Buf, y: gdnProjBuf,
                        m: UInt32(D), n: UInt32(gdnInternal))
            qwenResidualAdd.encode(commandBuffer: cb,
                                   hidden: hidden, delta: gdnProjBuf,
                                   count: Int(D))
        }

        // FFN: shared expert + routed MoE
        try runSync { cb in
            rms.encodeBF16W(commandBuffer: cb, x: hidden,
                            weight: postAttn.buffer, weightOffset: Int(postAttn.offset),
                            out: denseX, d: D, eps: eps)
            rms.encodeBF16W(commandBuffer: cb, x: hidden,
                            weight: postAttn.buffer, weightOffset: Int(postAttn.offset),
                            out: routedX, d: D, eps: eps)
            rms.encodeBF16W(commandBuffer: cb, x: hidden,
                            weight: postAttn.buffer, weightOffset: Int(postAttn.offset),
                            out: routerInput, d: D, eps: eps)
        }

        try runSync { cb in
            let inputScale = 1.0 / sqrt(Float(cfg.hiddenSize))
            qwenRouter.encode(commandBuffer: cb,
                              logits: routerInput,
                              outIndices: outIndices, outWeights: outWeights,
                              numExperts: UInt32(cfg.numExperts), topK: UInt32(cfg.topKExperts),
                              inputScale: inputScale, tokenCount: 1)
        }

        let idxPtr = outIndices.contents().bindMemory(to: UInt32.self, capacity: cfg.topKExperts)
        var experts = [Int](repeating: 0, count: cfg.topKExperts)
        for i in 0..<cfg.topKExperts { experts[i] = min(Int(idxPtr[i]), cfg.numExperts - 1) }

        let routedOffsets = model.routedExpertOffsets(layer: L)
        let topK = UInt32(cfg.topKExperts)

        let sharedCB = ctx.queue.makeCommandBuffer()!
        try shared.encode(commandBuffer: sharedCB,
                          x: denseX, gate: proj.gate, up: proj.up, down: proj.down,
                          y: h1Buf,
                          scratchGate: denseScratchGate, scratchUp: denseScratchUp,
                          scratchAct: denseScratchAct)
        sharedCB.commit()

        let blobs = try await model.fetchRoutedExperts(layer: L, experts: experts)
        let routedBufs = blobs.map { $0.buffer }

        try runSync { cb in
            let argBuf = moe.makeReusedRoutedArgumentBuffer(routedBlobs: routedBufs, topK: topK)
            moe.encodeRoutedPersistentPhase1U16Load(
                commandBuffer: cb, routedArgBuffer: argBuf, routedBlobs: routedBufs,
                routedOffsets: routedOffsets, x: routedX, acts: moeActs,
                d: D, f: UInt32(cfg.moeIntermediateSize), topK: topK)
            moe.encodeRoutedPersistentPhase2Reduce(
                commandBuffer: cb, routedArgBuffer: argBuf, routedBlobs: routedBufs,
                routedOffsets: routedOffsets, acts: moeActs, routingWeights: outWeights,
                residual: zeroResidual, y: h2Buf,
                d: D, f: UInt32(cfg.moeIntermediateSize), topK: topK)

            qwenFFNCombine.encode(commandBuffer: cb,
                                  hidden: hidden, routed: h2Buf, shared: h1Buf,
                                  gate: self.gateVecHalf[L],
                                  d: Int(D), count: Int(D))
        }
    }

    private func executePrefillChunk(tokens: ArraySlice<Int32>,
                                     startPosition: Int,
                                     outputMode: PrefillOutputMode,
                                     logits: MTLBuffer,
                                     scratch: PrefillChunkScratchBuffers,
                                     config: PrefillRuntimeConfig,
                                     writeFinalHead: Bool) async throws {
        guard !tokens.isEmpty else { return }
        guard let kv else {
            throw PrefillError.chunkedUnsupported("chunked prefill requires FP16 KV")
        }
        let kvPosition = kv.position
        guard kvPosition == startPosition else {
            throw PrefillError.chunkedUnsupported(
                "prefill cursor \(kvPosition) != startPosition \(startPosition)")
        }

        let D = cfg.hiddenSize
        let eps: Float = 1e-6
        let sqrtHidden = Float(D).squareRoot()
        let t = tokens.count
        let tokenIDs = tokens.map { UInt32(bitPattern: $0) }
        guard let tokenBuffer = ctx.device.makeBuffer(bytes: tokenIDs,
                                                      length: tokenIDs.count * MemoryLayout<UInt32>.stride,
                                                      options: .storageModeShared) else {
            throw ModelError.residentBufferWrapFailed
        }
        let emb = model.embedding
        let isFullAttnLayer: (Int) -> Bool = { L in self.cfg.fullAttentionLayerMask[L] != 0 }

        prefillChunkState.markDirty(startPosition: startPosition, tokenCount: t)

        // Embed
        guard let cb = ctx.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        prefillEmbed.encode(commandBuffer: cb,
                            table: emb.buffer, tableOffset: Int(emb.offset),
                            scales: emb.buffer, scalesOffset: Int(emb.scaleOffset),
                            biases: emb.buffer, biasesOffset: Int(emb.biasOffset),
                            tokens: tokenBuffer, out: scratch.hidden,
                            t: UInt32(t), d: UInt32(D), outScale: sqrtHidden)
        cb.commit()
        try wait(cb)
        dumpStats(scratch.hidden, rows: t, cols: D, label: "after_embed")
        dumpRaw(scratch.hidden, rows: t, cols: D, label: "embed")

        for L in 0..<cfg.numLayers {
            var layerCB = ctx.queue.makeCommandBuffer()!
            let isFull = isFullAttnLayer(L)
            if isFull {
                try await executePrefillFullAttnLayer(
                    L, cb: &layerCB, startPosition: startPosition, t: t, D: D, eps: eps,
                    scratch: scratch, logits: logits, outputMode: outputMode,
                    writeFinalHead: writeFinalHead && L == cfg.numLayers - 1, kv: kv)
            } else {
                try await executePrefillGDNLayer(
                    L, cb: &layerCB, startPosition: startPosition, t: t, D: D, eps: eps,
                    scratch: scratch, logits: logits, outputMode: outputMode,
                    writeFinalHead: writeFinalHead && L == cfg.numLayers - 1, kv: kv)
            }
            dumpStats(scratch.hidden, rows: 1, cols: D, label: "layer_\(L)")
            if L == 0 {
                dumpRaw(scratch.q, rows: t, cols: 8192, label: "mixed0")
                dumpRaw(scratch.gdnProj, rows: t, cols: D, label: "projout0")
                dumpRaw(scratch.h1, rows: t, cols: D, label: "shared0")
                dumpRaw(scratch.h2, rows: t, cols: D, label: "routed0")
                dumpRaw(scratch.hidden, rows: t, cols: D, label: "hidden0")
            }
            if L == 3 {
                dumpRaw(scratch.q, rows: t, cols: 8192, label: "L3_qraw")
                dumpRaw(scratch.qCompact, rows: t, cols: 4096, label: "L3_qcompact")
                dumpRaw(scratch.kStage, rows: t, cols: 512, label: "L3_kraw")
                dumpRaw(scratch.vStage, rows: t, cols: 512, label: "L3_vraw")
                dumpRaw(scratch.attentionOutput, rows: t, cols: 4096, label: "L3_attn")
                dumpRaw(scratch.hidden, rows: t, cols: D, label: "L3_hidden")
            }
            if L == 0 {
            }
            if !isFull {
                dumpStats(scratch.q, rows: 1, cols: cfg.linearNumKeyHeads * cfg.linearHeadDim * 2 + cfg.linearNumValueHeads * cfg.linearHeadDim, label: "L\(L)_mixed")
                dumpStats(scratch.kStage, rows: 1, cols: cfg.linearNumValueHeads * cfg.linearHeadDim, label: "L\(L)_z")
                dumpStats(scratch.h1, rows: 1, cols: D, label: "L\(L)_gdnout")
                dumpStats(scratch.gdnProj, rows: 1, cols: D, label: "L\(L)_projout")
                dumpStats(scratch.denseX, rows: 1, cols: D, label: "L\(L)_denseX")
                dumpStats(scratch.h2, rows: 1, cols: D, label: "L\(L)_routed")
            }
        }

        kv.advance(by: t)
        if writeFinalHead {
            dumpStats(scratch.hidden, rows: t, cols: D, label: "pre_head")
        }
        prefillChunkState.markCommitted()
    }

    private func executePrefillFullAttnLayer(
        _ L: Int, cb: inout MTLCommandBuffer, startPosition: Int, t: Int,
        D: Int, eps: Float, scratch: PrefillChunkScratchBuffers,
        logits: MTLBuffer, outputMode: PrefillOutputMode,
        writeFinalHead: Bool, kv: KVCacheManager) async throws {

        let headDimL = cfg.headDim
        let numKVL = cfg.numKVHeads
        let qDim = cfg.numHeads * headDimL
        let kvDim = numKVL * headDimL
        let inNorm = try model.inputNorm(layer: L)
        let qW = try model.qProj(layer: L)
        let kW = try model.kProj(layer: L)
        let vW = try model.vProj(layer: L)
        let oW = try model.oProj(layer: L)
        let qNormV = try model.qNorm(layer: L)
        let kNormV = try model.kNorm(layer: L)
        func encodeInt4Proj(_ weights: TensorView, _ y: MTLBuffer,
                             _ rows: Int, _ cols: Int) {
            prefillQMM.encode(commandBuffer: cb,
                              weights: weights.buffer, weightsOffset: Int(weights.offset),
                              scales: weights.buffer, scalesOffset: Int(weights.scaleOffset),
                              biases: weights.buffer, biasesOffset: Int(weights.biasOffset),
                              x: scratch.normed, y: y, t: t, n: rows, k: cols)
        }

        prefillRMS.encodeBF16W(commandBuffer: cb, x: scratch.hidden,
                               weight: inNorm.buffer, weightOffset: Int(inNorm.offset),
                               out: scratch.normed, t: UInt32(t), d: UInt32(D), eps: eps)
        if L == 3 {
            dumpRaw(scratch.normed, rows: t, cols: D, label: "L3_normed")
        }

        encodeInt4Proj(qW, scratch.q, qDim * 2, D)
        encodeInt4Proj(kW, scratch.kStage, kvDim, D)
        encodeInt4Proj(vW, scratch.vStage, kvDim, D)

        // q/k epilogue: per-head RMSNorm + partial-segment RoPE, writing
        // the roped K directly into the KV cache and the compact q into qCompact.
        let keyBuf = kv.keyBuffer(layer: L, validTokenCount: startPosition + t)
        let valBuf = kv.valueBuffer(layer: L, validTokenCount: startPosition + t)
        let rotatedPairs = UInt32(Double(headDimL) * cfg.partialRotaryFactor / 2.0)
        qwenAttnQK.encode(commandBuffer: cb,
                          qRaw: scratch.q, kRaw: scratch.kStage,
                          qWeight: qNormV.buffer, qWeightOffset: Int(qNormV.offset),
                          kWeight: kNormV.buffer, kWeightOffset: Int(kNormV.offset),
                          qOut: scratch.qCompact, kOut: keyBuf,
                          kOutOffset: startPosition * kvDim * 2,
                          headDim: UInt32(headDimL), numQHeads: UInt32(cfg.numHeads),
                          numKVHeads: UInt32(numKVL), rotatedPairs: rotatedPairs,
                          theta: Float(cfg.ropeTheta), positionBase: UInt32(startPosition),
                          tokenCount: UInt32(t), eps: eps)

        // write V into the KV cache (values are not roped)
        if let blit = cb.makeBlitCommandEncoder() {
            blit.copy(from: scratch.vStage, sourceOffset: 0,
                      to: valBuf, destinationOffset: startPosition * kvDim * 2,
                      size: t * kvDim * 2)
            blit.endEncoding()
        }

        let params = PrefillAttentionParams(
            startPosition: UInt32(startPosition), queryCount: UInt32(t),
            headDim: UInt32(headDimL), numQHeads: UInt32(cfg.numHeads),
            numKVHeads: UInt32(numKVL),
            kvValidCount: UInt32(startPosition + t),
            slidingWindow: UInt32(cfg.slidingWindow),
            kvTokenStrideElements: UInt32(kvDim),
            qTokenStrideElements: UInt32(qDim),
            oTokenStrideElements: UInt32(qDim), scale: 1.0 / sqrt(Float(headDimL)))
        prefillAttention.encodeCausal(commandBuffer: cb,
                                      q: scratch.qCompact, k: keyBuf, v: valBuf,
                                      out: scratch.attentionOutput, params: params)

        qwenGateMul.encode(commandBuffer: cb,
                           attn: scratch.attentionOutput, qRaw: scratch.q,
                           headDim: UInt32(headDimL),
                           numQHeads: UInt32(cfg.numHeads),
                           tokenCount: UInt32(t))

        encodeInt4Proj(oW, scratch.h1, D, qDim)

        qwenResidualAdd.encode(commandBuffer: cb,
                               hidden: scratch.hidden, delta: scratch.h1,
                               count: t * D)

        // Post-attention norm (single Qwen3.5 norm) for all FFN inputs
        let postAttn = try model.postAttnNorm(layer: L)
        prefillRMS.encodeBF16W(commandBuffer: cb, x: scratch.hidden,
                               weight: postAttn.buffer, weightOffset: Int(postAttn.offset),
                               out: scratch.denseX, t: UInt32(t), d: UInt32(D), eps: eps)
        prefillRMS.encodeBF16W(commandBuffer: cb, x: scratch.hidden,
                               weight: postAttn.buffer, weightOffset: Int(postAttn.offset),
                               out: scratch.routedX, t: UInt32(t), d: UInt32(D), eps: eps)
        prefillRMS.encodeBF16W(commandBuffer: cb, x: scratch.hidden,
                               weight: postAttn.buffer, weightOffset: Int(postAttn.offset),
                               out: scratch.routerX, t: UInt32(t), d: UInt32(D), eps: eps)

        // Router
        let routerW = try model.qwenRouter(layer: L)
        prefillQMM.encode(commandBuffer: cb,
                          weights: routerW.buffer, weightsOffset: Int(routerW.offset),
                          scales: routerW.buffer, scalesOffset: Int(routerW.scaleOffset),
                          biases: routerW.buffer, biasesOffset: Int(routerW.biasOffset),
                          x: scratch.routerX, y: scratch.routePartials,
                          t: t, n: cfg.numExperts, k: D)

        cb.commit()
        try wait(cb)

        let routeCount = t * cfg.topKExperts
        let routerInputScale = 1.0 / sqrt(Float(cfg.hiddenSize))
        guard let routerCB = ctx.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        qwenRouter.encode(commandBuffer: routerCB,
                          logits: scratch.routePartials,
                          outIndices: scratch.routeIDs, outWeights: scratch.routeWeights,
                          numExperts: UInt32(cfg.numExperts), topK: UInt32(cfg.topKExperts),
                          inputScale: routerInputScale, tokenCount: UInt32(t))
        routerCB.commit()
        try wait(routerCB)

        let idPtr = scratch.routeIDs.contents().bindMemory(to: UInt32.self, capacity: routeCount)
        let weightPtr = scratch.routeWeights.contents().bindMemory(to: Float16.self, capacity: routeCount)
        var routeIDs = [UInt32]()
        var routeWeights = [Float16]()
        routeIDs.reserveCapacity(routeCount)
        routeWeights.reserveCapacity(routeCount)
        for i in 0..<routeCount {
            routeIDs.append(min(idPtr[i], UInt32(cfg.numExperts - 1)))
            routeWeights.append(weightPtr[i])
        }
        let pairs = PrefillRouter.makeTokenExpertPairs(indices: routeIDs,
                                                       weights: routeWeights,
                                                       queryCount: t,
                                                       topK: cfg.topKExperts)
        let schedulerConfig = Self.prefillRoutedTileSchedulerConfig
        let routeTileExpertCount: Int
        if let slotCount = model.routedExpertCacheSlotCount(layer: L) {
            guard schedulerConfig.fitsSlotBudget(slotCount: slotCount) else {
                throw PrefillError.chunkedUnsupported(
                    "prefill routed tile depth \(schedulerConfig.maxPendingDepth) needs \((schedulerConfig.maxPendingDepth + 1) * schedulerConfig.tileExperts) slots, has \(slotCount)")
            }
            routeTileExpertCount = min(schedulerConfig.tileExperts, slotCount)
        } else {
            routeTileExpertCount = schedulerConfig.tileExperts
        }
        let routes = try PrefillMoEGrouping.groupTokenExpertPairs(
            pairs, queryCount: t, topK: cfg.topKExperts,
            numExperts: cfg.numExperts, tileExpertCount: routeTileExpertCount,
            expertSortKeys: model.routedExpertPhysicalOffsets(layer: L))

        // Shared expert (parallel with routed I/O)
        let proj = sharedExpertProjections[L]
        guard let sharedCB = ctx.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        try prefillSharedExpert.encodeBlock(commandBuffer: sharedCB,
                                            x: scratch.denseX, y: scratch.h1,
                                            gate: proj.gate, up: proj.up, down: proj.down,
                                            scratchGate: scratch.sharedGateScratch,
                                            scratchUp: scratch.sharedUpScratch,
                                            scratchAct: scratch.sharedActScratch,
                                            queryCount: t, d: D,
                                            intermediate: cfg.moeIntermediateSize,
                                            xStrideElements: D, yStrideElements: D)
        sharedCB.commit()
        try wait(sharedCB)

        // Routed expert streaming
        let metadata = try prefillGroupedMoE.makeStreamedMetadataBuffers(
            device: ctx.device, routes: routes)
        let routedOffsets = model.routedExpertOffsets(layer: L)

        struct PendingPrefillTile {
            let tileIndex: Int
            let commandBuffer: MTLCommandBuffer
            let fetch: PrefillStreamedTileFetchResult
            let argumentBuffer: PrefillStreamedTileArgumentBuffer
        }
        var pendingTiles: [PendingPrefillTile] = []
        var tileLifetime = PrefillStreamedTileSlotLifetime()

        func drainOldestPendingTile() throws {
            guard !pendingTiles.isEmpty else { return }
            let pending = pendingTiles.removeFirst()
            try withExtendedLifetime((pending.fetch, pending.argumentBuffer)) {
                try wait(pending.commandBuffer)
            }
            if !pending.fetch.plannedMissSlots.isEmpty {
                try tileLifetime.complete(tileIndex: pending.tileIndex)
            }
        }

        let routedTileScheduler = PrefillRoutedTileScheduler(config: schedulerConfig)
        for (tileIndex, tile) in routes.tiles.enumerated() {
            let expertIDs = try PrefillStreamedTileBinding.expertIDs(
                forTile: tileIndex, routes: routes)
            var plannedFetch: RoutedExpertFetchPlan?
            if !pendingTiles.isEmpty {
                let pendingAssignedSlots = pendingTiles.flatMap(\.fetch.plannedAssignedSlots)
                if !pendingAssignedSlots.isEmpty {
                    let pendingSlots = Set(pendingAssignedSlots)
                    let plan = try model.planRoutedExpertsIfPossible(
                        layer: L, experts: expertIDs, avoidingSlots: pendingSlots)
                    let decision = routedTileScheduler.decide(
                        PrefillRoutedTileSchedulerInput(
                            hasPendingTile: true,
                            pendingDepth: pendingTiles.count,
                            pendingAssignedSlots: pendingAssignedSlots,
                            avoidingSlotPlanAvailable: plan != nil))
                    switch decision {
                    case .prefetchNext:
                        guard let plan else {
                            throw ModelError.indexCorrupt(
                                detail: "routed tile scheduler requested missing plan")
                        }
                        plannedFetch = plan
                    case .drainBeforeIssue:
                        try drainOldestPendingTile()
                    case .issueWithoutPending:
                        throw ModelError.indexCorrupt(
                            detail: "routed tile scheduler ignored pending tile")
                    }
                } else {
                    let decision = routedTileScheduler.decide(
                        PrefillRoutedTileSchedulerInput(
                            hasPendingTile: true,
                            pendingDepth: pendingTiles.count,
                            pendingAssignedSlots: [],
                            avoidingSlotPlanAvailable: false))
                    switch decision {
                    case .drainBeforeIssue:
                        try drainOldestPendingTile()
                    case .issueWithoutPending, .prefetchNext:
                        throw ModelError.indexCorrupt(
                            detail: "routed tile scheduler failed to drain empty-slot pending tile")
                    }
                }
            } else {
                let decision = routedTileScheduler.decide(
                    PrefillRoutedTileSchedulerInput(
                        hasPendingTile: false,
                        pendingAssignedSlots: [],
                        avoidingSlotPlanAvailable: false))
                switch decision {
                case .issueWithoutPending: break
                case .prefetchNext, .drainBeforeIssue:
                    throw ModelError.indexCorrupt(
                        detail: "routed tile scheduler requested pending action without pending tile")
                }
            }
            let fetch = try await PrefillStreamedTileBinding.fetchBindingForTile(
                model: model, layer: L, tileIndex: tileIndex, routes: routes,
                plannedFetch: plannedFetch,
                avoidingSlots: Set(pendingTiles.flatMap(\.fetch.plannedAssignedSlots)))
            try fetch.binding.validateCoversPairs(routes.sortedPairs,
                                                  pairStart: Int(tile.pairStart),
                                                  pairCount: Int(tile.pairCount))
            if !fetch.plannedMissSlots.isEmpty {
                try tileLifetime.begin(tileIndex: tileIndex,
                                       plannedSlots: fetch.plannedMissSlots)
            }
            let argumentBuffer = try prefillGroupedMoE.makeStreamedArgumentBuffer(
                device: ctx.device, binding: fetch.binding)
            let streamedParams = PrefillGroupedRoutedMoEStreamedParams(
                pairStart: tile.pairStart, pairCount: tile.pairCount,
                d: UInt32(D), routedIntermediate: UInt32(cfg.moeIntermediateSize),
                topK: UInt32(cfg.topKExperts), hiddenStrideElements: UInt32(D),
                binding: fetch.binding, offsets: routedOffsets)
            guard let tileCB = ctx.queue.makeCommandBuffer() else {
                throw ModelError.residentBufferWrapFailed
            }
            _ = prefillGroupedMoE.encodeStreamedBatched(
                commandBuffer: tileCB,
                hidden: scratch.routedX,
                sortedPairs: metadata.sortedPairs,
                routePartials: scratch.routePartials,
                gateUpActScratch: scratch.routedGateUpActScratch,
                downScratch: scratch.routedDownScratch,
                argumentBuffer: argumentBuffer,
                binding: fetch.binding,
                params: streamedParams,
                pairMicrobatchRows: scratch.layout.routedPairMicrobatchRows)
            tileCB.commit()
            pendingTiles.append(PendingPrefillTile(tileIndex: tileIndex,
                                                   commandBuffer: tileCB,
                                                   fetch: fetch,
                                                   argumentBuffer: argumentBuffer))
            while pendingTiles.count > schedulerConfig.maxPendingDepth {
                try drainOldestPendingTile()
            }
        }
        while !pendingTiles.isEmpty {
            try drainOldestPendingTile()
        }

        // Reduce routed experts and combine: hidden += h2 + h1 * sigmoid(gate)
        
        guard let tailCB = ctx.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        prefillMoE.encodeReduceTokenMajor(commandBuffer: tailCB,
                                          routePartials: scratch.routePartials,
                                          routeWeights: scratch.routeWeights,
                                          h2: scratch.h2,
                                          queryCount: UInt32(t),
                                          topK: UInt32(cfg.topKExperts),
                                          d: UInt32(D))
        qwenFFNCombine.encode(commandBuffer: tailCB,
                              hidden: scratch.hidden, routed: scratch.h2,
                              shared: scratch.h1, gate: self.gateVecHalf[L],
                              d: D, count: t * D)
        tailCB.commit()
        try withExtendedLifetime(metadata) { try wait(tailCB) }

        if writeFinalHead {
            let fn = model.finalNorm
            let lm = model.lmHead
            guard let finalCB = ctx.queue.makeCommandBuffer() else {
                throw ModelError.residentBufferWrapFailed
            }
            dumpRaw(scratch.hidden, rows: t, cols: D, label: "final_hidden")
            prefillFinalRowHead.encodeLogits(commandBuffer: finalCB,
                                             hiddenBlock: scratch.hidden,
                                             row: t - 1, rowStrideElements: D,
                                             normWeight: fn.buffer,
                                             normWeightOffset: Int(fn.offset),
                                             weights: lm.buffer, weightsOffset: Int(lm.offset),
                                             scales: lm.buffer, scalesOffset: Int(lm.scaleOffset),
                                             biases: lm.buffer, biasesOffset: Int(lm.biasOffset),
                                             logits: logits,
                                             d: UInt32(D), vocab: UInt32(cfg.vocabSize),
                                             rmsEps: eps)
            finalCB.commit()
            try wait(finalCB)
        }
    }

    private func executePrefillGDNLayer(
        _ L: Int, cb: inout MTLCommandBuffer, startPosition: Int, t: Int,
        D: Int, eps: Float, scratch: PrefillChunkScratchBuffers,
        logits: MTLBuffer, outputMode: PrefillOutputMode,
        writeFinalHead: Bool, kv: KVCacheManager) async throws {

        let inNorm = try model.inputNorm(layer: L)
        let inQKV = try model.inProjQKV(layer: L)
        let inZ = try model.inProjZ(layer: L)
        let inA = try model.inProjA(layer: L)
        let inB = try model.inProjB(layer: L)
        let convW = try model.gdnConvWeight(layer: L)
        let aLogV = try model.gdnALog(layer: L)
        let dtBiasV = try model.gdnDtBias(layer: L)
        let normWV = try model.gdnNormWeight(layer: L)
        let outProj = try model.qwenGDNOutProj(layer: L)
        let convDim = cfg.linearNumKeyHeads * cfg.linearHeadDim * 2
            + cfg.linearNumValueHeads * cfg.linearHeadDim

        // Each GDN layer keeps its own recurrent state in gdnState[L] /
        // gdnConvState[L]; it persists across chunks (reset only on reset()).

        dumpStatsBF16(normWV.buffer, cols: cfg.linearHeadDim, label: "L\(L)_normw",
                      offsetElems: Int(normWV.offset))

        prefillRMS.encodeBF16W(commandBuffer: cb, x: scratch.hidden,
                               weight: inNorm.buffer, weightOffset: Int(inNorm.offset),
                               out: scratch.normed, t: UInt32(t), d: UInt32(D), eps: eps)
        prefillQMM.encode(commandBuffer: cb,
                          weights: inQKV.buffer, weightsOffset: Int(inQKV.offset),
                          scales: inQKV.buffer, scalesOffset: Int(inQKV.scaleOffset),
                          biases: inQKV.buffer, biasesOffset: Int(inQKV.biasOffset),
                           x: scratch.normed, y: scratch.q,
                           t: t, n: convDim, k: D)
        prefillQMM.encode(commandBuffer: cb,
                           weights: inZ.buffer, weightsOffset: Int(inZ.offset),
                          scales: inZ.buffer, scalesOffset: Int(inZ.scaleOffset),
                          biases: inZ.buffer, biasesOffset: Int(inZ.biasOffset),
                           x: scratch.normed, y: scratch.kStage,
                           t: t, n: cfg.linearNumValueHeads * cfg.linearHeadDim, k: D)
        prefillQMM.encode(commandBuffer: cb,
                           weights: inA.buffer, weightsOffset: Int(inA.offset),
                           scales: inA.buffer, scalesOffset: Int(inA.scaleOffset),
                           biases: inA.buffer, biasesOffset: Int(inA.biasOffset),
                           x: scratch.normed, y: scratch.routeWeights,
                           t: t, n: cfg.linearNumValueHeads, k: D)
        prefillQMM.encode(commandBuffer: cb,
                           weights: inB.buffer, weightsOffset: Int(inB.offset),
                           scales: inB.buffer, scalesOffset: Int(inB.scaleOffset),
                           biases: inB.buffer, biasesOffset: Int(inB.biasOffset),
                           x: scratch.normed, y: scratch.routePartials,
                           t: t, n: cfg.linearNumValueHeads, k: D)

        let inputScale = 1.0 / sqrt(Float(cfg.linearHeadDim))
        qwenGDN.encode(commandBuffer: cb,
                       mixed: scratch.q, z: scratch.kStage,
                       aVec: scratch.routeWeights, bVec: scratch.routePartials,
                       convWeight: convW.buffer, convWeightOffset: Int(convW.offset),
                        convState: gdnConvState[L], state: gdnState[L],
                       aLog: aLogV.buffer, aLogOffset: Int(aLogV.offset),
                       dtBias: dtBiasV.buffer, dtBiasOffset: Int(dtBiasV.offset),
                       normWeight: normWV.buffer, normWeightOffset: Int(normWV.offset),
                       out: scratch.h1,
                       tokenCount: UInt32(t),
                       valueHeads: UInt32(cfg.linearNumValueHeads),
                       keyHeads: UInt32(cfg.linearNumKeyHeads),
                        convDim: UInt32(convDim),
                        kernelDim: UInt32(cfg.linearConvKernelDim),
                        eps: eps, inputScale: inputScale)

        // GDN out_proj: gated per-head output (num_value_heads*value_head_dim)
        // -> hidden, then residual add.
        let gdnInternal = cfg.linearNumValueHeads * cfg.linearHeadDim
        prefillQMM.encode(commandBuffer: cb,
                          weights: outProj.buffer, weightsOffset: Int(outProj.offset),
                          scales: outProj.buffer, scalesOffset: Int(outProj.scaleOffset),
                          biases: outProj.buffer, biasesOffset: Int(outProj.biasOffset),
                          x: scratch.h1, y: scratch.gdnProj,
                          t: t, n: D, k: gdnInternal)

        qwenResidualAdd.encode(commandBuffer: cb,
                               hidden: scratch.hidden, delta: scratch.gdnProj,
                               count: t * D)

        // Post-GDN norm (single Qwen3.5 norm) for all FFN inputs
        let postAttn = try model.postAttnNorm(layer: L)
        prefillRMS.encodeBF16W(commandBuffer: cb, x: scratch.hidden,
                               weight: postAttn.buffer, weightOffset: Int(postAttn.offset),
                               out: scratch.denseX, t: UInt32(t), d: UInt32(D), eps: eps)
        prefillRMS.encodeBF16W(commandBuffer: cb, x: scratch.hidden,
                               weight: postAttn.buffer, weightOffset: Int(postAttn.offset),
                               out: scratch.routedX, t: UInt32(t), d: UInt32(D), eps: eps)
        prefillRMS.encodeBF16W(commandBuffer: cb, x: scratch.hidden,
                               weight: postAttn.buffer, weightOffset: Int(postAttn.offset),
                               out: scratch.routerX, t: UInt32(t), d: UInt32(D), eps: eps)

        // Router
        let routerW = try model.qwenRouter(layer: L)
        prefillQMM.encode(commandBuffer: cb,
                          weights: routerW.buffer, weightsOffset: Int(routerW.offset),
                          scales: routerW.buffer, scalesOffset: Int(routerW.scaleOffset),
                          biases: routerW.buffer, biasesOffset: Int(routerW.biasOffset),
                          x: scratch.routerX, y: scratch.routePartials,
                          t: t, n: cfg.numExperts, k: D)

        cb.commit()
        try wait(cb)

        let routeCount = t * cfg.topKExperts
        let routerInputScale = 1.0 / sqrt(Float(cfg.hiddenSize))
        guard let routerCB = ctx.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        qwenRouter.encode(commandBuffer: routerCB,
                          logits: scratch.routePartials,
                          outIndices: scratch.routeIDs, outWeights: scratch.routeWeights,
                          numExperts: UInt32(cfg.numExperts), topK: UInt32(cfg.topKExperts),
                          inputScale: routerInputScale, tokenCount: UInt32(t))
        routerCB.commit()
        try wait(routerCB)

        let idPtr = scratch.routeIDs.contents().bindMemory(to: UInt32.self, capacity: routeCount)
        let weightPtr = scratch.routeWeights.contents().bindMemory(to: Float16.self, capacity: routeCount)
        var routeIDs = [UInt32]()
        var routeWeights = [Float16]()
        routeIDs.reserveCapacity(routeCount)
        routeWeights.reserveCapacity(routeCount)
        for i in 0..<routeCount {
            routeIDs.append(min(idPtr[i], UInt32(cfg.numExperts - 1)))
            routeWeights.append(weightPtr[i])
        }
        let pairs = PrefillRouter.makeTokenExpertPairs(indices: routeIDs,
                                                       weights: routeWeights,
                                                       queryCount: t,
                                                       topK: cfg.topKExperts)
        let schedulerConfig = Self.prefillRoutedTileSchedulerConfig
        let routeTileExpertCount: Int
        if let slotCount = model.routedExpertCacheSlotCount(layer: L) {
            guard schedulerConfig.fitsSlotBudget(slotCount: slotCount) else {
                throw PrefillError.chunkedUnsupported(
                    "prefill routed tile depth \(schedulerConfig.maxPendingDepth) needs \((schedulerConfig.maxPendingDepth + 1) * schedulerConfig.tileExperts) slots, has \(slotCount)")
            }
            routeTileExpertCount = min(schedulerConfig.tileExperts, slotCount)
        } else {
            routeTileExpertCount = schedulerConfig.tileExperts
        }
        let routes = try PrefillMoEGrouping.groupTokenExpertPairs(
            pairs, queryCount: t, topK: cfg.topKExperts,
            numExperts: cfg.numExperts, tileExpertCount: routeTileExpertCount,
            expertSortKeys: model.routedExpertPhysicalOffsets(layer: L))

        // Shared expert (parallel with routed I/O)
        let proj = sharedExpertProjections[L]
        guard let sharedCB = ctx.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        try prefillSharedExpert.encodeBlock(commandBuffer: sharedCB,
                                            x: scratch.denseX, y: scratch.h1,
                                            gate: proj.gate, up: proj.up, down: proj.down,
                                            scratchGate: scratch.sharedGateScratch,
                                            scratchUp: scratch.sharedUpScratch,
                                            scratchAct: scratch.sharedActScratch,
                                            queryCount: t, d: D,
                                            intermediate: cfg.moeIntermediateSize,
                                            xStrideElements: D, yStrideElements: D)
        sharedCB.commit()
        try wait(sharedCB)

        // Routed expert streaming
        let metadata = try prefillGroupedMoE.makeStreamedMetadataBuffers(
            device: ctx.device, routes: routes)
        let routedOffsets = model.routedExpertOffsets(layer: L)

        struct PendingPrefillTile {
            let tileIndex: Int
            let commandBuffer: MTLCommandBuffer
            let fetch: PrefillStreamedTileFetchResult
            let argumentBuffer: PrefillStreamedTileArgumentBuffer
        }
        var pendingTiles: [PendingPrefillTile] = []
        var tileLifetime = PrefillStreamedTileSlotLifetime()

        func drainOldestPendingTile() throws {
            guard !pendingTiles.isEmpty else { return }
            let pending = pendingTiles.removeFirst()
            try withExtendedLifetime((pending.fetch, pending.argumentBuffer)) {
                try wait(pending.commandBuffer)
            }
            if !pending.fetch.plannedMissSlots.isEmpty {
                try tileLifetime.complete(tileIndex: pending.tileIndex)
            }
        }

        let routedTileScheduler = PrefillRoutedTileScheduler(config: schedulerConfig)
        for (tileIndex, tile) in routes.tiles.enumerated() {
            let expertIDs = try PrefillStreamedTileBinding.expertIDs(
                forTile: tileIndex, routes: routes)
            var plannedFetch: RoutedExpertFetchPlan?
            if !pendingTiles.isEmpty {
                let pendingAssignedSlots = pendingTiles.flatMap(\.fetch.plannedAssignedSlots)
                if !pendingAssignedSlots.isEmpty {
                    let pendingSlots = Set(pendingAssignedSlots)
                    let plan = try model.planRoutedExpertsIfPossible(
                        layer: L, experts: expertIDs, avoidingSlots: pendingSlots)
                    let decision = routedTileScheduler.decide(
                        PrefillRoutedTileSchedulerInput(
                            hasPendingTile: true,
                            pendingDepth: pendingTiles.count,
                            pendingAssignedSlots: pendingAssignedSlots,
                            avoidingSlotPlanAvailable: plan != nil))
                    switch decision {
                    case .prefetchNext:
                        guard let plan else {
                            throw ModelError.indexCorrupt(
                                detail: "routed tile scheduler requested missing plan")
                        }
                        plannedFetch = plan
                    case .drainBeforeIssue:
                        try drainOldestPendingTile()
                    case .issueWithoutPending:
                        throw ModelError.indexCorrupt(
                            detail: "routed tile scheduler ignored pending tile")
                    }
                } else {
                    let decision = routedTileScheduler.decide(
                        PrefillRoutedTileSchedulerInput(
                            hasPendingTile: true,
                            pendingDepth: pendingTiles.count,
                            pendingAssignedSlots: [],
                            avoidingSlotPlanAvailable: false))
                    switch decision {
                    case .drainBeforeIssue:
                        try drainOldestPendingTile()
                    case .issueWithoutPending, .prefetchNext:
                        throw ModelError.indexCorrupt(
                            detail: "routed tile scheduler failed to drain empty-slot pending tile")
                    }
                }
            } else {
                let decision = routedTileScheduler.decide(
                    PrefillRoutedTileSchedulerInput(
                        hasPendingTile: false,
                        pendingAssignedSlots: [],
                        avoidingSlotPlanAvailable: false))
                switch decision {
                case .issueWithoutPending: break
                case .prefetchNext, .drainBeforeIssue:
                    throw ModelError.indexCorrupt(
                        detail: "routed tile scheduler requested pending action without pending tile")
                }
            }
            let fetch = try await PrefillStreamedTileBinding.fetchBindingForTile(
                model: model, layer: L, tileIndex: tileIndex, routes: routes,
                plannedFetch: plannedFetch,
                avoidingSlots: Set(pendingTiles.flatMap(\.fetch.plannedAssignedSlots)))
            try fetch.binding.validateCoversPairs(routes.sortedPairs,
                                                  pairStart: Int(tile.pairStart),
                                                  pairCount: Int(tile.pairCount))
            if !fetch.plannedMissSlots.isEmpty {
                try tileLifetime.begin(tileIndex: tileIndex,
                                       plannedSlots: fetch.plannedMissSlots)
            }
            let argumentBuffer = try prefillGroupedMoE.makeStreamedArgumentBuffer(
                device: ctx.device, binding: fetch.binding)
            let streamedParams = PrefillGroupedRoutedMoEStreamedParams(
                pairStart: tile.pairStart, pairCount: tile.pairCount,
                d: UInt32(D), routedIntermediate: UInt32(cfg.moeIntermediateSize),
                topK: UInt32(cfg.topKExperts), hiddenStrideElements: UInt32(D),
                binding: fetch.binding, offsets: routedOffsets)
            guard let tileCB = ctx.queue.makeCommandBuffer() else {
                throw ModelError.residentBufferWrapFailed
            }
            _ = prefillGroupedMoE.encodeStreamedBatched(
                commandBuffer: tileCB,
                hidden: scratch.routedX,
                sortedPairs: metadata.sortedPairs,
                routePartials: scratch.routePartials,
                gateUpActScratch: scratch.routedGateUpActScratch,
                downScratch: scratch.routedDownScratch,
                argumentBuffer: argumentBuffer,
                binding: fetch.binding,
                params: streamedParams,
                pairMicrobatchRows: scratch.layout.routedPairMicrobatchRows)
            tileCB.commit()
            pendingTiles.append(PendingPrefillTile(tileIndex: tileIndex,
                                                   commandBuffer: tileCB,
                                                   fetch: fetch,
                                                   argumentBuffer: argumentBuffer))
            while pendingTiles.count > schedulerConfig.maxPendingDepth {
                try drainOldestPendingTile()
            }
        }
        while !pendingTiles.isEmpty {
            try drainOldestPendingTile()
        }

        // Reduce routed experts and combine: hidden += h2 + h1 * sigmoid(gate)
        
        guard let tailCB = ctx.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        prefillMoE.encodeReduceTokenMajor(commandBuffer: tailCB,
                                          routePartials: scratch.routePartials,
                                          routeWeights: scratch.routeWeights,
                                          h2: scratch.h2,
                                          queryCount: UInt32(t),
                                          topK: UInt32(cfg.topKExperts),
                                          d: UInt32(D))
        qwenFFNCombine.encode(commandBuffer: tailCB,
                              hidden: scratch.hidden, routed: scratch.h2,
                              shared: scratch.h1, gate: self.gateVecHalf[L],
                              d: D, count: t * D)
        tailCB.commit()
        try withExtendedLifetime(metadata) { try wait(tailCB) }

        if writeFinalHead {
            let fn = model.finalNorm
            let lm = model.lmHead
            guard let finalCB = ctx.queue.makeCommandBuffer() else {
                throw ModelError.residentBufferWrapFailed
            }
            dumpRaw(scratch.hidden, rows: t, cols: D, label: "final_hidden")
            prefillFinalRowHead.encodeLogits(commandBuffer: finalCB,
                                             hiddenBlock: scratch.hidden,
                                             row: t - 1, rowStrideElements: D,
                                             normWeight: fn.buffer,
                                             normWeightOffset: Int(fn.offset),
                                             weights: lm.buffer, weightsOffset: Int(lm.offset),
                                             scales: lm.buffer, scalesOffset: Int(lm.scaleOffset),
                                             biases: lm.buffer, biasesOffset: Int(lm.biasOffset),
                                             logits: logits,
                                             d: UInt32(D), vocab: UInt32(cfg.vocabSize),
                                             rmsEps: eps)
            finalCB.commit()
            try wait(finalCB)
        }
    }
}
