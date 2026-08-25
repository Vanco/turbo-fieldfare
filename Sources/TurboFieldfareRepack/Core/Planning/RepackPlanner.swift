import Foundation
import TurboFieldfareFormat

/// On-disk page alignment unit for `.gturbo` files. Fixed at 16 KB regardless
/// of host page size — the format is the contract, not the kernel.
enum Layout {
    static let pageBytes = GTurboFormatV1.alignmentBytes
}

// MARK: - Plan data types

struct ResidentEntry: Sendable {
    let name: String
    /// dtype byte for IndexEntry: 0 = U32, 1 = BF16, 2 = FP16, 3 = FP32.
    let dtype: UInt8
    /// Logical shape after dequant (max rank 4; trailing zeros).
    let logicalShape4: [UInt32]
    /// File offset where the (packed) weight bytes start.
    let fileOffset: UInt64
    /// Size in bytes of the weight bytes.
    let sizeBytes: UInt64
    /// Offset where BF16 scales start (0 if none).
    let scaleOffset: UInt64
    let scaleSize: UInt64
    /// Offset where BF16 biases start (0 if none).
    let biasOffset: UInt64
    let biasSize: UInt64
    /// Quantization spec (nil for unquantized scalars/norms).
    let quantSpec: QuantSpec?

    /// Source tensors that supply this entry's bytes.
    let sourceWeight: SourceTensor
    let sourceScales: SourceTensor?
    let sourceBiases: SourceTensor?

    /// Precomputed bytes for tensors forced to a fixed dtype (e.g. Qwen GDN
    /// conv1d/A_log/dt_bias). When non-nil the writer emits these directly as
    /// an unquantized tensor and ignores the source tensors.
    let precomputed: Data?

    init(name: String, dtype: UInt8, logicalShape4: [UInt32], fileOffset: UInt64,
         sizeBytes: UInt64, scaleOffset: UInt64, scaleSize: UInt64,
         biasOffset: UInt64, biasSize: UInt64, quantSpec: QuantSpec?,
         sourceWeight: SourceTensor, sourceScales: SourceTensor?,
         sourceBiases: SourceTensor?, precomputed: Data? = nil) {
        self.name = name
        self.dtype = dtype
        self.logicalShape4 = logicalShape4
        self.fileOffset = fileOffset
        self.sizeBytes = sizeBytes
        self.scaleOffset = scaleOffset
        self.scaleSize = scaleSize
        self.biasOffset = biasOffset
        self.biasSize = biasSize
        self.quantSpec = quantSpec
        self.sourceWeight = sourceWeight
        self.sourceScales = sourceScales
        self.sourceBiases = sourceBiases
        self.precomputed = precomputed
    }
}

struct ResidentFilePlan: Sendable {
    let path: String
    let entries: [ResidentEntry]
    let stringTable: [UInt8]
    let stringTableOffsets: [UInt32]   // per-entry offsets into the table
    let indexSize: UInt64              // header + entries + table + padding
    let residentSize: UInt64           // tensor payload region
    var totalSize: UInt64 { indexSize + residentSize }
}

struct PerExpertTensorSlice: Sendable {
    let role: String                   // "gate" | "up" | "down"
    let component: String              // "weights" | "scales" | "biases"
    let dtype: UInt8                   // 0=U32, 1=BF16
    let logicalShape: [UInt64]         // per-expert logical shape
    let offsetInExpertBlob: UInt64     // within each expert blob
    let sizeInExpertBlob: UInt64
    /// For each expert e (0..<expertsPerLayer): source byte offset & size.
    let sourceOffsetPerExpert: UInt64  // stride per expert in source
    let sourceTensor: SourceTensor
    let bitsForWeights: Int?           // 4 for routed expert weight; nil for scales/biases
}

struct LayerFilePlan: Sendable {
    let layerIndex: Int
    let path: String
    let expertsPerLayer: Int
    let expertStride: UInt64
    let subTensors: [PerExpertTensorSlice]  // 9 entries: gate/up/down × {weights, scales, biases}
    var fileSize: UInt64 { UInt64(expertsPerLayer) * expertStride }

    func physicalRank(for logicalExpert: Int) -> Int {
        logicalExpert
    }

    init(layerIndex: Int,
                path: String,
                expertsPerLayer: Int,
                expertStride: UInt64,
                subTensors: [PerExpertTensorSlice]) {
        self.layerIndex = layerIndex
        self.path = path
        self.expertsPerLayer = expertsPerLayer
        self.expertStride = expertStride
        self.subTensors = subTensors
    }
}

struct RepackPlan: Sendable {
    let arch: ArchInfo
    let baseMode: String                  // "affine"
    let baseGroupSize: Int                // 64
    let bitsOverrideCount: Int
    let resident: ResidentFilePlan
    let layers: [LayerFilePlan]
    let matchedModelID: String?
    let excludedMultimodalTensorNames: [String]
}

// MARK: - Planner

enum RepackPlanner {

    /// Classify a tensor name. Routed-expert tensors split off the LM bucket.
    enum Bucket: Equatable {
        case lmResident
        case routedExpert(role: String, layer: Int)   // role = "gate"|"up"|"down"
        case excludedMultimodal
        case unknown
    }

    static func classify(_ name: String, numLayers: Int) -> Bucket {
        if name.hasPrefix("language_model.") {
            // Routed expert?
            if let role = routedExpertRole(in: name),
               let layer = layerIndex(in: name),
               layer >= 0 && layer < numLayers {
                return .routedExpert(role: role, layer: layer)
            }
            return .lmResident
        }
        if isMultimodalTensorName(name) {
            return .excludedMultimodal
        }
        return .unknown
    }

    private static func routedExpertRole(in name: String) -> String? {
        // 检查是否包含 ".mlp.switch_mlp."（Qwen）或 ".experts.switch_glu."（Gemma）
        guard name.contains(".mlp.switch_mlp.") || name.contains(".experts.switch_glu.") else {
            return nil
        }
        if name.contains(".gate_proj.") { return "gate" }
        if name.contains(".up_proj.")   { return "up" }
        if name.contains(".down_proj.") { return "down" }
        return nil
    }

    private static func layerIndex(in name: String) -> Int? {
        // matches "...layers.<N>...."
        guard let r = name.range(of: ".layers.") else { return nil }
        let tail = name[r.upperBound...]
        guard let dot = tail.firstIndex(of: ".") else { return nil }
        return Int(tail[tail.startIndex..<dot])
    }

    /// Build the plan from parsed shard headers + source metadata.
    /// - throws: classification + companion + override count failures.
    static func plan(meta: IndexLoader.SourceMetadata,
                            arch: ArchInfo,
                            shardHeaders: [Safetensors.Header],
                            outputDir: String,
                            remote: HuggingFaceRemoteSource? = nil,
                            remoteFiles: [String: RemoteFileInfo] = [:]) async throws -> RepackPlan {

        // Companion tensors may live in different shards, so resolve them
        // through one global registry.
        var registry: [String: SourceTensor] = [:]
        registry.reserveCapacity(meta.weightMap.count)
        for h in shardHeaders {
            for t in h.tensors { registry[t.name] = t }
        }

        // Source allowlisting owns exact fingerprint validation. Preserve the
        // declared override count for the output manifest audit.
        let bitsOverrideCount = meta.bitsOverrides.count

        var lmResidentBases: [String] = []
        var excludedMultimodalNames: [String] = []
        var routedByLayerAndRole: [Int: [String: String]] = [:]
        for (name, _) in registry {
            if isMultimodalTensorName(name) {
                excludedMultimodalNames.append(name)
            }
            if name.hasSuffix(".scales") || name.hasSuffix(".biases") { continue }
            let b = classify(name, numLayers: arch.numLayers)
            switch b {
            case .lmResident:                   lmResidentBases.append(name)
            case .routedExpert(let role, let layer):
                var byRole = routedByLayerAndRole[layer] ?? [:]
                if byRole[role] != nil {
                    throw RepackError.configurationInvalid(detail:
                        "two routed-expert tensors for layer \(layer) role \(role)")
                }
                byRole[role] = name
                routedByLayerAndRole[layer] = byRole
            case .excludedMultimodal:           continue
            case .unknown:                      throw RepackError.unknownTensorPrefix(name: name)
            }
        }

        // Sort deterministically. The LM order follows a fixed template.
        lmResidentBases.sort(by: lmResidentOrdering())
        excludedMultimodalNames.sort()

        let residentPath = (outputDir as NSString).appendingPathComponent("model_weights.bin")
        let resident = try await planResidentFile(path: residentPath,
                                             baseNames: lmResidentBases,
                                             registry: registry, meta: meta,
                                             remote: remote, remoteFiles: remoteFiles)

        let layersDir = (outputDir as NSString).appendingPathComponent("packed_experts")
        var layerPlans: [LayerFilePlan] = []
        layerPlans.reserveCapacity(arch.numLayers)
        for layer in 0..<arch.numLayers {
            let bundle = routedByLayerAndRole[layer] ?? [:]
            // Synthetic snapshots may legitimately have no routed experts.
            guard let gName = bundle["gate"], let uName = bundle["up"], let dName = bundle["down"] else {
                if bundle.isEmpty {
                    layerPlans.append(LayerFilePlan(layerIndex: layer,
                                                    path: (layersDir as NSString).appendingPathComponent("layer_\(String(format: "%02d", layer)).bin"),
                                                    expertsPerLayer: 0,
                                                    expertStride: 0,
                                                    subTensors: []))
                    continue
                }
                throw RepackError.configurationInvalid(detail:
                    "layer \(layer) routed-expert bundle incomplete: \(bundle)")
            }
            let path = (layersDir as NSString)
                .appendingPathComponent("layer_\(String(format: "%02d", layer)).bin")
            let lp = try planLayerFile(path: path, layer: layer,
                                       gateName: gName, upName: uName, downName: dName,
                                       registry: registry, meta: meta, arch: arch)
            layerPlans.append(lp)
        }

        let matched = SourceFingerprint.modelID(forIndexSha256: meta.indexSha256Hex)

        return RepackPlan(arch: arch,
                          baseMode: meta.baseMode,
                          baseGroupSize: meta.baseGroupSize,
                          bitsOverrideCount: bitsOverrideCount,
                          resident: resident,
                          layers: layerPlans,
                          matchedModelID: matched,
                          excludedMultimodalTensorNames: excludedMultimodalNames)
    }

    private static func isMultimodalTensorName(_ name: String) -> Bool {
        name.hasPrefix("vision_tower.") ||
            name.hasPrefix("embed_vision.") ||
            name.hasPrefix("audio_tower.")
    }

    // MARK: - Resident planning

    private static func planResidentFile(path: String,
                                          baseNames: [String],
                                          registry: [String: SourceTensor],
                                          meta: IndexLoader.SourceMetadata,
                                          remote: HuggingFaceRemoteSource? = nil,
                                          remoteFiles: [String: RemoteFileInfo] = [:]) async throws
                                         -> ResidentFilePlan {
        let entryCount = baseNames.count

        var stringTable: [UInt8] = []
        var offsets: [UInt32] = []
        offsets.reserveCapacity(entryCount)
        for n in baseNames {
            offsets.append(UInt32(stringTable.count))
            stringTable.append(contentsOf: n.utf8)
        }

        // Index size includes the fixed header, fixed-width entries, and the
        // string table, padded to a 16 KB page boundary.
        let rawIdx = UInt64(GTurboBinary.indexHeaderBytes
            + entryCount * GTurboBinary.indexEntryBytes
            + stringTable.count)
        let indexSize = roundUpToPage(rawIdx)

        var fileCursor = indexSize
        var entries: [ResidentEntry] = []
        entries.reserveCapacity(entryCount)

        for name in baseNames {
            guard let weight = registry[name] else {
                throw RepackError.missingTensor(name: name)
            }
            let dtype = ietnyDtype(weight.dtype)
            let isQuantizedPacked = (weight.dtype == .u32) && name.hasSuffix(".weight")

            if let target = forcedTargetDType(name) {
                // Qwen GDN tensors the runtime requires at a fixed precision
                // (BF16 for conv1d/dt_bias/norm, FP32 for A_log). The source may
                // store them quantized; dequantize (or convert) to the target
                // dtype and keep them unquantized so requireBF16/requireFP32 pass.
                let logical: [UInt64]
                let data: Data
                if isQuantizedPacked {
                    let base = String(name.dropLast(".weight".count))
                    guard let scales = registry[base + ".scales"] else {
                        throw RepackError.missingScalesCompanion(name: name)
                    }
                    guard let biases = registry[base + ".biases"] else {
                        throw RepackError.missingBiasesCompanion(name: name)
                    }
                    if scales.dtype != .bf16 || biases.dtype != .bf16 {
                        throw RepackError.dtypeMismatch(name: name,
                            detail: "expected BF16 scales/biases, got \(scales.dtype)/\(biases.dtype)")
                    }
                    let spec = IndexLoader.quantSpec(forTensor: name, meta: meta)
                    logical = logicalShape(forPackedSource: weight.shape, bits: spec.bits)
                    data = try await dequantizeResident(name: name, weight: weight, scales: scales,
                                                  biases: biases, logical: logical,
                                                  target: target, meta: meta,
                                                  remote: remote, remoteFiles: remoteFiles)
                } else {
                    logical = weight.shape
                    data = try await convertResident(name: name, weight: weight,
                                               logical: logical, target: target,
                                               remote: remote, remoteFiles: remoteFiles)
                }
                let off = (fileCursor + 3) & ~UInt64(3)
                let size = UInt64(data.count)
                fileCursor = off + size
                entries.append(ResidentEntry(
                    name: name, dtype: target, logicalShape4: padTo4(truncateTrailingOnes(logical)),
                    fileOffset: off, sizeBytes: size,
                    scaleOffset: 0, scaleSize: 0, biasOffset: 0, biasSize: 0,
                    quantSpec: nil, sourceWeight: weight, sourceScales: nil,
                    sourceBiases: nil, precomputed: data))
            } else if isQuantizedPacked {
                let base = String(name.dropLast(".weight".count))
                guard let scales = registry[base + ".scales"] else {
                    throw RepackError.missingScalesCompanion(name: name)
                }
                guard let biases = registry[base + ".biases"] else {
                    throw RepackError.missingBiasesCompanion(name: name)
                }
                if scales.dtype != .bf16 || biases.dtype != .bf16 {
                    throw RepackError.dtypeMismatch(name: name,
                        detail: "expected BF16 scales/biases, got \(scales.dtype)/\(biases.dtype)")
                }
                let spec = IndexLoader.quantSpec(forTensor: name, meta: meta)
                let logical = logicalShape(forPackedSource: weight.shape, bits: spec.bits)

                let wOff = fileCursor
                let wSize = weight.sizeBytes
                let sOff = wOff + wSize
                let sSize = scales.sizeBytes
                let bOff = sOff + sSize
                let bSize = biases.sizeBytes
                fileCursor = bOff + bSize

                entries.append(ResidentEntry(
                    name: name, dtype: GTurboFormatV1.DType.u32.rawValue,
                    logicalShape4: padTo4(logical),
                    fileOffset: wOff, sizeBytes: wSize,
                    scaleOffset: sOff, scaleSize: sSize,
                    biasOffset: bOff, biasSize: bSize,
                    quantSpec: spec,
                    sourceWeight: weight, sourceScales: scales, sourceBiases: biases))
            } else {
                // Unquantized (BF16 norm / scalar) — no companions.
                let off = fileCursor
                let size = weight.sizeBytes
                fileCursor = off + size

                entries.append(ResidentEntry(
                    name: name, dtype: dtype,
                    logicalShape4: padTo4(weight.shape),
                    fileOffset: off, sizeBytes: size,
                    scaleOffset: 0, scaleSize: 0,
                    biasOffset: 0, biasSize: 0,
                    quantSpec: nil,
                    sourceWeight: weight, sourceScales: nil, sourceBiases: nil))
            }
        }

        let residentSize = fileCursor - indexSize

        return ResidentFilePlan(path: path,
                                entries: entries,
                                stringTable: stringTable,
                                stringTableOffsets: offsets,
                                indexSize: indexSize,
                                residentSize: residentSize)
    }

    // MARK: - Layer planning

    private static func planLayerFile(path: String, layer: Int,
                                      gateName: String, upName: String, downName: String,
                                      registry: [String: SourceTensor],
                                      meta: IndexLoader.SourceMetadata,
                                      arch: ArchInfo) throws -> LayerFilePlan {
        let expertCount = arch.numExperts
        let roles: [(role: String, name: String)] = [
            ("gate", gateName), ("up", upName), ("down", downName)
        ]
        var subs: [PerExpertTensorSlice] = []
        subs.reserveCapacity(9)
        var blobCursor: UInt64 = 0

        for (role, name) in roles {
            guard let w = registry[name] else { throw RepackError.missingTensor(name: name) }
            if w.dtype != .u32 || w.shape.count != 3 || Int(w.shape[0]) != expertCount {
                throw RepackError.shapeMismatch(name: name,
                    detail: "expected U32 rank-3 with leading \(expertCount), got \(w.dtype) \(w.shape)")
            }
            let base = name.hasSuffix(".weight") ? String(name.dropLast(".weight".count)) : name
            guard let s = registry[base + ".scales"] else { throw RepackError.missingScalesCompanion(name: name) }
            guard let b = registry[base + ".biases"] else { throw RepackError.missingBiasesCompanion(name: name) }
            if s.dtype != .bf16 || b.dtype != .bf16 {
                throw RepackError.dtypeMismatch(name: name,
                    detail: "expected BF16 scales/biases, got \(s.dtype)/\(b.dtype)")
            }

            let perExpertWeightSize = w.sizeBytes / UInt64(expertCount)
            let perExpertScaleSize  = s.sizeBytes / UInt64(expertCount)
            let perExpertBiasSize   = b.sizeBytes / UInt64(expertCount)
            if perExpertWeightSize * UInt64(expertCount) != w.sizeBytes ||
               perExpertScaleSize  * UInt64(expertCount) != s.sizeBytes ||
               perExpertBiasSize   * UInt64(expertCount) != b.sizeBytes {
                throw RepackError.shapeMismatch(name: name,
                    detail: "source bytes not evenly divisible by \(expertCount) experts")
            }

            let spec = IndexLoader.quantSpec(forTensor: name, meta: meta)
            let perExpertSourceShape = Array(w.shape.dropFirst())
            let logicalPerExpert = logicalShape(forPackedSource: perExpertSourceShape, bits: spec.bits)
            let scalesLogical = Array(s.shape.dropFirst())
            let biasesLogical = Array(b.shape.dropFirst())

            let wSlice = PerExpertTensorSlice(
                role: role, component: "weights", dtype: GTurboFormatV1.DType.u32.rawValue,
                logicalShape: logicalPerExpert,
                offsetInExpertBlob: blobCursor, sizeInExpertBlob: perExpertWeightSize,
                sourceOffsetPerExpert: perExpertWeightSize, sourceTensor: w,
                bitsForWeights: spec.bits)
            blobCursor += perExpertWeightSize
            let sSlice = PerExpertTensorSlice(
                role: role, component: "scales", dtype: GTurboFormatV1.DType.bf16.rawValue,
                logicalShape: scalesLogical,
                offsetInExpertBlob: blobCursor, sizeInExpertBlob: perExpertScaleSize,
                sourceOffsetPerExpert: perExpertScaleSize, sourceTensor: s,
                bitsForWeights: nil)
            blobCursor += perExpertScaleSize
            let bSlice = PerExpertTensorSlice(
                role: role, component: "biases", dtype: GTurboFormatV1.DType.bf16.rawValue,
                logicalShape: biasesLogical,
                offsetInExpertBlob: blobCursor, sizeInExpertBlob: perExpertBiasSize,
                sourceOffsetPerExpert: perExpertBiasSize, sourceTensor: b,
                bitsForWeights: nil)
            blobCursor += perExpertBiasSize

            subs.append(wSlice); subs.append(sSlice); subs.append(bSlice)
        }

        let expertStride = roundUpToPage(blobCursor)
        return LayerFilePlan(layerIndex: layer, path: path,
                             expertsPerLayer: expertCount,
                             expertStride: expertStride,
                             subTensors: subs)
    }

    // MARK: - Helpers

    private static func ietnyDtype(_ d: SourceTensor.Dtype) -> UInt8 {
        switch d { case .u32: 0; case .bf16: 1; case .fp16: 2; case .fp32: 3 }
    }

    private static func roundUpToPage(_ v: UInt64) -> UInt64 {
        let p = Layout.pageBytes
        return ((v + p - 1) / p) * p
    }

    private static func padTo4(_ s: [UInt64]) -> [UInt32] {
        var out: [UInt32] = []
        out.reserveCapacity(4)
        for v in s.prefix(4) { out.append(UInt32(v)) }
        while out.count < 4 { out.append(0) }
        return out
    }

    /// Drops trailing size-1 dimensions so a tensor like `[8192, 4, 1]` is
    /// recorded as `[8192, 4]`, matching the 2-D schema the runtime expects for
    /// GDN conv1d weights. Total element count is unchanged.
    private static func truncateTrailingOnes(_ s: [UInt64]) -> [UInt64] {
        var out = s
        while out.count > 1, let last = out.last, last == 1 {
            out.removeLast()
        }
        return out
    }

    /// Logical shape of a packed quantized tensor whose source is `[D0,..,Dn-1, Dn/factor]`.
    private static func logicalShape(forPackedSource source: [UInt64], bits: Int) -> [UInt64] {
        let factor = UInt64(32 / bits)
        guard !source.isEmpty else { return source }
        var out = source
        out[out.count - 1] = source[source.count - 1] * factor
        return out
    }

    // MARK: - Forced-precision tensors (Qwen GDN)

    /// Tensors the runtime demands at a fixed precision regardless of how the
    /// source quantized them. Returns the target `GTurboFormatV1.DType` byte, or
    /// nil to let the normal quant/unquant paths handle the tensor.
    private static func forcedTargetDType(_ name: String) -> UInt8? {
        if name.hasSuffix(".linear_attn.A_log") {
            return GTurboFormatV1.DType.fp32.rawValue
        }
        if name.hasSuffix(".linear_attn.conv1d.weight")
            || name.hasSuffix(".linear_attn.dt_bias")
            || name.hasSuffix(".linear_attn.norm.weight") {
            return GTurboFormatV1.DType.bf16.rawValue
        }
        return nil
    }

    private static func readSourceBytes(_ t: SourceTensor,
                                         remote: HuggingFaceRemoteSource?,
                                         remoteFiles: [String: RemoteFileInfo]) async throws -> Data {
        if let remote, let info = remoteFiles[t.shardPath] {
            let tmp = try await remote.pinned(commit: info.resolvedCommit)
                .downloadRangeToTempFile(filename: t.shardPath, info: info,
                                         offset: t.absoluteOffset, length: Int(t.sizeBytes))
            defer { try? FileManager.default.removeItem(atPath: tmp.path) }
            let data = try Data(contentsOf: URL(fileURLWithPath: tmp.path))
            guard data.count == Int(t.sizeBytes) else {
                throw RepackError.remoteBodyTruncated(
                    path: t.shardPath, expected: t.sizeBytes, actual: UInt64(data.count))
            }
            return data
        }
        let handle = try MmapHandle(path: t.shardPath)
        let slice = handle.slice(at: t.absoluteOffset, count: Int(t.sizeBytes))
        return Data(bytes: slice.baseAddress!, count: Int(t.sizeBytes))
    }

    private static func dequantizeResident(name: String, weight: SourceTensor,
                                           scales: SourceTensor, biases: SourceTensor,
                                           logical: [UInt64], target: UInt8,
                                           meta: IndexLoader.SourceMetadata,
                                           remote: HuggingFaceRemoteSource?,
                                           remoteFiles: [String: RemoteFileInfo]) async throws -> Data {
        _ = meta
        let wBytes = try await readSourceBytes(weight, remote: remote, remoteFiles: remoteFiles)
        let sBytes = try await readSourceBytes(scales, remote: remote, remoteFiles: remoteFiles)
        let bBytes = try await readSourceBytes(biases, remote: remote, remoteFiles: remoteFiles)
        let logicalCount = Int(logical.reduce(1, *))
        let targetFP32 = target == GTurboFormatV1.DType.fp32.rawValue
        return dequantizeInt4ToFloating(weightBytes: wBytes, scalesBytes: sBytes,
                                        biasesBytes: bBytes, logicalCount: logicalCount,
                                        targetFP32: targetFP32)
    }

    private static func convertResident(name: String, weight: SourceTensor,
                                        logical: [UInt64], target: UInt8,
                                        remote: HuggingFaceRemoteSource?,
                                        remoteFiles: [String: RemoteFileInfo]) async throws -> Data {
        _ = logical
        let src = try await readSourceBytes(weight, remote: remote, remoteFiles: remoteFiles)
        let count = src.count / weight.dtype.elementBytes
        let targetFP32 = target == GTurboFormatV1.DType.fp32.rawValue
        return convertFloating(srcData: src, srcDType: weight.dtype, count: count,
                                targetFP32: targetFP32)
    }

    /// Dequantize unsigned int4 (low nibble of byte k = element 2k, high nibble
    /// = element 2k+1; value = nibble*scale + bias) to BF16 or FP32, matching
    /// the runtime convention in `dequant_int4.metal`.
    private static func dequantizeInt4ToFloating(weightBytes: Data,
                                                 scalesBytes: Data,
                                                 biasesBytes: Data,
                                                 logicalCount: Int,
                                                 targetFP32: Bool) -> Data {
        let groupSize = 64
        if targetFP32 {
            var out = Data(count: logicalCount * 4)
            out.withUnsafeMutableBytes { ob in
                let o = ob.bindMemory(to: Float.self)
                weightBytes.withUnsafeBytes { wb in
                    scalesBytes.withUnsafeBytes { sb in
                        biasesBytes.withUnsafeBytes { bb in
                            let w = wb.bindMemory(to: UInt8.self)
                            let s = sb.bindMemory(to: UInt16.self)
                            let b = bb.bindMemory(to: UInt16.self)
                            for i in 0..<logicalCount {
                                let byte = w[i / 2]
                                let nibble = (i & 1) == 0 ? UInt32(byte & 0x0F) : UInt32(byte >> 4)
                                let g = i / groupSize
                                let scale = bf16ToFloat(s[g])
                                let bias = bf16ToFloat(b[g])
                                o[i] = Float(nibble) * scale + bias
                            }
                        }
                    }
                }
            }
            return out
        } else {
            var out = Data(count: logicalCount * 2)
            out.withUnsafeMutableBytes { ob in
                let o = ob.bindMemory(to: UInt16.self)
                weightBytes.withUnsafeBytes { wb in
                    scalesBytes.withUnsafeBytes { sb in
                        biasesBytes.withUnsafeBytes { bb in
                            let w = wb.bindMemory(to: UInt8.self)
                            let s = sb.bindMemory(to: UInt16.self)
                            let b = bb.bindMemory(to: UInt16.self)
                            for i in 0..<logicalCount {
                                let byte = w[i / 2]
                                let nibble = (i & 1) == 0 ? UInt32(byte & 0x0F) : UInt32(byte >> 4)
                                let g = i / groupSize
                                let scale = bf16ToFloat(s[g])
                                let bias = bf16ToFloat(b[g])
                                let value = Float(nibble) * scale + bias
                                o[i] = floatToBF16(value)
                            }
                        }
                    }
                }
            }
            return out
        }
    }

    /// Reinterpret already-unquantized floating source bytes into BF16 or FP32.
    private static func convertFloating(srcData: Data, srcDType: SourceTensor.Dtype,
                                        count: Int, targetFP32: Bool) -> Data {
        if targetFP32 {
            var out = Data(count: count * 4)
            out.withUnsafeMutableBytes { ob in
                let o = ob.bindMemory(to: Float.self)
                srcData.withUnsafeBytes { sb in
                    for i in 0..<count {
                        o[i] = floatValue(srcData: sb, srcDType: srcDType, index: i)
                    }
                }
            }
            return out
        } else {
            var out = Data(count: count * 2)
            out.withUnsafeMutableBytes { ob in
                let o = ob.bindMemory(to: UInt16.self)
                srcData.withUnsafeBytes { sb in
                    for i in 0..<count {
                        o[i] = floatToBF16(floatValue(srcData: sb, srcDType: srcDType, index: i))
                    }
                }
            }
            return out
        }
    }

    private static func floatValue(srcData sb: UnsafeRawBufferPointer,
                                   srcDType: SourceTensor.Dtype, index: Int) -> Float {
        guard let base = sb.baseAddress else { return 0 }
        switch srcDType {
        case .bf16:
            let p = base.advanced(by: index * 2).assumingMemoryBound(to: UInt16.self)
            return bf16ToFloat(p.pointee)
        case .fp16:
            let p = base.advanced(by: index * 2).assumingMemoryBound(to: UInt16.self)
            return Float(Float16(bitPattern: p.pointee))
        case .fp32:
            let p = base.advanced(by: index * 4).assumingMemoryBound(to: Float.self)
            return p.pointee
        default:
            return 0
        }
    }

    private static func bf16ToFloat(_ h: UInt16) -> Float {
        Float(bitPattern: UInt32(h) << 16)
    }

    private static func floatToBF16(_ f: Float) -> UInt16 {
        let bits = f.bitPattern
        let low = bits & 0xFFFF
        var result = UInt16((bits >> 16) & 0xFFFF)
        if low > 0x8000 || (low == 0x8000 && (result & 1) != 0) {
            result &+= 1
        }
        return result
    }

    /// Stable order for the resident LM tensor list. Embedding first, then
    /// per-layer groups in layer index order, then the final norm.
    private static func lmResidentOrdering() -> (String, String) -> Bool {
        // Compute a sort key per name; we order by (group rank, layer, slot rank, name).
        func key(_ n: String) -> (Int, Int, Int, String) {
            if n == "language_model.model.embed_tokens.weight" { return (0, 0, 0, n) }
            if n == "language_model.model.norm.weight"          { return (3, 0, 0, n) }
            if let li = layerIndex(in: n) {
                let slot = slotRank(in: n)
                return (1, li, slot, n)
            }
            return (2, 0, 0, n)
        }
        return { a, b in
            let ka = key(a), kb = key(b)
            if ka.0 != kb.0 { return ka.0 < kb.0 }
            if ka.1 != kb.1 { return ka.1 < kb.1 }
            if ka.2 != kb.2 { return ka.2 < kb.2 }
            return ka.3 < kb.3
        }
    }

    /// Within-layer slot order. Mirrors the per-layer description in the
    /// architecture doc.
    private static func slotRank(in n: String) -> Int {
        if n.contains(".self_attn.q_proj.weight") { return 0 }
        if n.contains(".self_attn.k_proj.weight") { return 1 }
        if n.contains(".self_attn.v_proj.weight") { return 2 }
        if n.contains(".self_attn.o_proj.weight") { return 3 }
        if n.contains(".self_attn.q_norm.weight") { return 4 }
        if n.contains(".self_attn.k_norm.weight") { return 5 }
        if n.contains(".router.proj.weight")      { return 6 }
        if n.contains(".router.scale")            { return 7 }
        if n.contains(".router.per_expert_scale") { return 8 }
        if n.contains(".mlp.gate_proj.weight")    { return 9 }
        if n.contains(".mlp.up_proj.weight")      { return 10 }
        if n.contains(".mlp.down_proj.weight")    { return 11 }
        if n.hasSuffix(".input_layernorm.weight") { return 12 }
        if n.hasSuffix(".post_attention_layernorm.weight") { return 13 }
        if n.hasSuffix(".pre_feedforward_layernorm.weight") { return 14 }
        if n.hasSuffix(".pre_feedforward_layernorm_2.weight") { return 15 }
        if n.hasSuffix(".post_feedforward_layernorm.weight") { return 16 }
        if n.hasSuffix(".post_feedforward_layernorm_1.weight") { return 17 }
        if n.hasSuffix(".post_feedforward_layernorm_2.weight") { return 18 }
        if n.hasSuffix(".layer_scalar")           { return 19 }
        return 100
    }
}
