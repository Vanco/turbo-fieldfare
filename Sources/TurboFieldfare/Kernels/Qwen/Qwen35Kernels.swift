import Foundation
import Metal

/// Swift wrappers for the Qwen 3.5 kernels in `Metal/Qwen/qwen.metal`.
/// All runtime-parameterized — the Qwen path does not use PSO specialization.

/// Per-head Q/K RMSNorm + partial-segment RoPE for output-gated attention.
/// Q raw layout per token: [head | gate] halves; only head halves are
/// normalized/roped into `qOut`. K is normalized + roped into `kOut`.
final class QwenAttnQKEpilogue {
    private let pso: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.pso = try context.pipeline("qwen_attn_qk_epilogue")
    }

    func encode(commandBuffer cb: MTLCommandBuffer,
                qRaw: MTLBuffer,
                kRaw: MTLBuffer,
                qWeight: MTLBuffer,
                qWeightOffset: Int,
                kWeight: MTLBuffer,
                kWeightOffset: Int,
                qOut: MTLBuffer,
                kOut: MTLBuffer,
                kOutOffset: Int,
                headDim: UInt32,
                numQHeads: UInt32,
                numKVHeads: UInt32,
                rotatedPairs: UInt32,
                theta: Float,
                positionBase: UInt32,
                tokenCount: UInt32,
                eps: Float) {
        let qInStride = numQHeads * 2 * headDim
        let kInStride = numKVHeads * headDim
        let qOutStride = numQHeads * headDim
        let kOutStride = numKVHeads * headDim
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        enc.setBuffer(qRaw, offset: 0, index: 0)
        enc.setBuffer(kRaw, offset: 0, index: 1)
        enc.setBuffer(qWeight, offset: qWeightOffset, index: 2)
        enc.setBuffer(kWeight, offset: kWeightOffset, index: 3)
        enc.setBuffer(qOut, offset: 0, index: 4)
        enc.setBuffer(kOut, offset: 0, index: 5)
        var headDimVar = headDim
        var nqVar = numQHeads
        var nkvVar = numKVHeads
        var rpVar = rotatedPairs
        var thetaVar = theta
        var posVar = positionBase
        var epsVar = eps
        var qInVar = qInStride
        var kInVar = kInStride
        var qOutVar = qOutStride
        var kOutVar = kOutStride
        var kOffVar = UInt32(kOutOffset)
        enc.setBytes(&headDimVar, length: MemoryLayout<UInt32>.size, index: 6)
        enc.setBytes(&nqVar,      length: MemoryLayout<UInt32>.size, index: 7)
        enc.setBytes(&nkvVar,     length: MemoryLayout<UInt32>.size, index: 8)
        enc.setBytes(&rpVar,      length: MemoryLayout<UInt32>.size, index: 9)
        enc.setBytes(&thetaVar,   length: MemoryLayout<Float>.size,  index: 10)
        enc.setBytes(&posVar,     length: MemoryLayout<UInt32>.size, index: 11)
        enc.setBytes(&epsVar,     length: MemoryLayout<Float>.size,  index: 12)
        enc.setBytes(&qInVar,     length: MemoryLayout<UInt32>.size, index: 13)
        enc.setBytes(&kInVar,     length: MemoryLayout<UInt32>.size, index: 14)
        enc.setBytes(&qOutVar,    length: MemoryLayout<UInt32>.size, index: 15)
        enc.setBytes(&kOutVar,    length: MemoryLayout<UInt32>.size, index: 16)
        enc.setBytes(&kOffVar,    length: MemoryLayout<UInt32>.size, index: 17)
        let threads = min(Int(pso.maxTotalThreadsPerThreadgroup), 256)
        let flat = (Int(numQHeads + numKVHeads)) * Int(tokenCount)
        enc.dispatchThreadgroups(
            MTLSize(width: flat, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        enc.endEncoding()
    }
}

/// attn *= sigmoid(gate half of qRaw), elementwise over [T, numQHeads*headDim].
final class QwenSigmoidGateMul {
    private let pso: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.pso = try context.pipeline("qwen_sigmoid_gate_mul")
    }

    func encode(commandBuffer cb: MTLCommandBuffer,
                attn: MTLBuffer,
                qRaw: MTLBuffer,
                headDim: UInt32,
                numQHeads: UInt32,
                tokenCount: UInt32) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        enc.setBuffer(attn, offset: 0, index: 0)
        enc.setBuffer(qRaw, offset: 0, index: 1)
        var hdVar = headDim
        var nqVar = numQHeads
        var tVar = tokenCount
        var strideVar = numQHeads * 2 * headDim
        enc.setBytes(&hdVar,     length: MemoryLayout<UInt32>.size, index: 2)
        enc.setBytes(&nqVar,     length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&tVar,      length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&strideVar, length: MemoryLayout<UInt32>.size, index: 5)
        let total = Int(tokenCount) * Int(numQHeads) * Int(headDim)
        let threads = min(Int(pso.maxTotalThreadsPerThreadgroup), 1024)
        let groups = (total + threads - 1) / threads
        enc.dispatchThreadgroups(
            MTLSize(width: max(groups, 1), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        enc.endEncoding()
    }
}

/// Softmax over all experts then top-k select; weights keep global probabilities.
final class QwenRouterSelect {
    private let pso: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.pso = try context.pipeline("qwen_router_topk_softmax_all")
    }

    func encode(commandBuffer cb: MTLCommandBuffer,
                logits: MTLBuffer,
                outIndices: MTLBuffer,
                outWeights: MTLBuffer,
                numExperts: UInt32,
                topK: UInt32,
                inputScale: Float,
                tokenCount: UInt32) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        enc.setBuffer(logits, offset: 0, index: 0)
        enc.setBuffer(outIndices, offset: 0, index: 1)
        enc.setBuffer(outWeights, offset: 0, index: 2)
        var neVar = numExperts
        var tkVar = topK
        var scaleVar = inputScale
        enc.setBytes(&neVar,    length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&tkVar,    length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&scaleVar, length: MemoryLayout<Float>.size,  index: 5)
        let threads = min(Int(pso.maxTotalThreadsPerThreadgroup),
                          max(Int(numExperts), 32))
        enc.dispatchThreadgroups(
            MTLSize(width: Int(tokenCount), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        enc.endEncoding()
    }
}

/// Fused gated-delta-net forward over T sequential tokens.
final class QwenGDNForward {
    private let pso: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.pso = try context.pipeline("qwen_gdn_forward")
    }

    func encode(commandBuffer cb: MTLCommandBuffer,
                mixed: MTLBuffer,
                z: MTLBuffer,
                aVec: MTLBuffer,
                bVec: MTLBuffer,
                convWeight: MTLBuffer,
                convWeightOffset: Int,
                convState: MTLBuffer,
                state: MTLBuffer,
                aLog: MTLBuffer,
                aLogOffset: Int,
                dtBias: MTLBuffer,
                dtBiasOffset: Int,
                normWeight: MTLBuffer,
                normWeightOffset: Int,
                out: MTLBuffer,
                tokenCount: UInt32,
                valueHeads: UInt32,
                keyHeads: UInt32,
                convDim: UInt32,
                kernelDim: UInt32,
                eps: Float,
                inputScale: Float) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        enc.setBuffer(mixed, offset: 0, index: 0)
        enc.setBuffer(z, offset: 0, index: 1)
        enc.setBuffer(aVec, offset: 0, index: 2)
        enc.setBuffer(bVec, offset: 0, index: 3)
        enc.setBuffer(convWeight, offset: convWeightOffset, index: 4)
        enc.setBuffer(convState, offset: 0, index: 5)
        enc.setBuffer(state, offset: 0, index: 6)
        enc.setBuffer(aLog, offset: aLogOffset, index: 7)
        enc.setBuffer(dtBias, offset: dtBiasOffset, index: 8)
        enc.setBuffer(normWeight, offset: normWeightOffset, index: 9)
        enc.setBuffer(out, offset: 0, index: 10)
        var tVar = tokenCount
        var vhVar = valueHeads
        var khVar = keyHeads
        var cdVar = convDim
        var kdVar = kernelDim
        var epsVar = eps
        var scaleVar = inputScale
        enc.setBytes(&tVar,     length: MemoryLayout<UInt32>.size, index: 11)
        enc.setBytes(&vhVar,    length: MemoryLayout<UInt32>.size, index: 12)
        enc.setBytes(&khVar,    length: MemoryLayout<UInt32>.size, index: 13)
        enc.setBytes(&cdVar,    length: MemoryLayout<UInt32>.size, index: 14)
        enc.setBytes(&kdVar,    length: MemoryLayout<UInt32>.size, index: 15)
        enc.setBytes(&epsVar,   length: MemoryLayout<Float>.size,  index: 16)
        enc.setBytes(&scaleVar, length: MemoryLayout<Float>.size,  index: 17)
        // One threadgroup per value head; the kernel body assumes exactly
        // linearHeadDim == 128 threads so every lane owns one column.
        let threads = 128
        enc.dispatchThreadgroups(
            MTLSize(width: Int(valueHeads), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        enc.endEncoding()
    }
}

/// hidden += delta, batched.
final class QwenResidualAddBatch {
    private let pso: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.pso = try context.pipeline("qwen_residual_add_batch")
    }

    func encode(commandBuffer cb: MTLCommandBuffer,
                hidden: MTLBuffer,
                delta: MTLBuffer,
                count: Int) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        enc.setBuffer(hidden, offset: 0, index: 0)
        enc.setBuffer(delta, offset: 0, index: 1)
        var countVar = UInt32(count)
        enc.setBytes(&countVar, length: MemoryLayout<UInt32>.size, index: 2)
        let threads = min(Int(pso.maxTotalThreadsPerThreadgroup), 1024)
        let groups = (count + threads - 1) / threads
        enc.dispatchThreadgroups(
            MTLSize(width: max(groups, 1), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        enc.endEncoding()
    }
}

/// hidden += routed + shared * sigmoid(scalar gate per token).
final class QwenFFNCombineBatch {
    private let pso: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.pso = try context.pipeline("qwen_ffn_combine_batch")
    }

    func encode(commandBuffer cb: MTLCommandBuffer,
                hidden: MTLBuffer,
                routed: MTLBuffer,
                shared: MTLBuffer,
                gate: MTLBuffer,
                d: Int,
                count: Int) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        enc.setBuffer(hidden, offset: 0, index: 0)
        enc.setBuffer(routed, offset: 0, index: 1)
        enc.setBuffer(shared, offset: 0, index: 2)
        enc.setBuffer(gate, offset: 0, index: 3)
        var dVar = UInt32(d)
        var countVar = UInt32(count)
        enc.setBytes(&dVar,     length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&countVar, length: MemoryLayout<UInt32>.size, index: 5)
        let threads = min(Int(pso.maxTotalThreadsPerThreadgroup), 1024)
        let groups = (count + threads - 1) / threads
        enc.dispatchThreadgroups(
            MTLSize(width: max(groups, 1), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        enc.endEncoding()
    }
}
