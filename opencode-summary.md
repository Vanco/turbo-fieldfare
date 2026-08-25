# Session Summary

## Objective
- Fully implement Qwen 3.5 35B-A3B support in turbo-fieldfare (Metal kernels, Swift wrappers, `Qwen35ForwardRunner`, and server wiring) so Qwen models load AND run end-to-end; fix `TurboFieldfareRepack` so it produces a correct Qwen `.gturbo` package (the existing `scratch/qwen35.gturbo` was packed before Qwen GDN handling existed and stores GDN tensors quantized, which the runtime rejects).

## Important Details
- **Qwen modelID** = `mlx-community/Qwen3.5-35B-A3B-4bit` (contains "qwen" → `ModelFamily.detect` returns `.qwen3_5_35B_A3B`). Manifest arch matches `ArchConfig.qwen3_5_35B_A3B` (hiddenSize=2048, numExperts=256, 40 layers, full-attn at indices 3,7,…39).
- **Qwen quant layout**: routed experts int4, shared expert + router **int8** (`manifest.quant.sharedExpert.weightBits = 8`, `router.weightBits = 8`). Runner must honor manifest, not hardcode 4.
- **Dequant convention** (`Sources/TurboFieldfare/Metal/Quant/dequant_int4.metal:8-12`): unsigned int4; low nibble of byte k = element 2k, high nibble = 2k+1; `value = float(nibble) * scale[i/64] + bias[i/64]`; scales/biases BF16; group size 64. `logicalShape(forPackedSource:bits:)` applies `factor = 32/bits` to the last dim (8 for int4).
- **GDN tensors the runtime REQUIRES at fixed precision** (`Model.swift` `validateRuntimeSchema` ~892-913): `linear_attn.conv1d.weight`→BF16, `linear_attn.A_log`→FP32, `linear_attn.dt_bias`→BF16, `linear_attn.norm.weight`→BF16 (all others like `in_proj_*`, `out_proj` are requireAffineInt4, fine).
- **Root cause of `resident index is corrupt: ...conv1d.weight does not match the required BF16 schema`**: `RepackPlanner.swift` `isQuantizedPacked = (weight.dtype == .u32) && name.hasSuffix(".weight")`. The "4bit" source stores GDN tensors as U32 (quantized), so the repack copied them quantized (with scales/biases); but runtime `requireBF16`/`requireFP32` demand unquantized. Repack had NO GDN awareness.
- `SourceTensor` (Safetensors.swift) exposes `shardPath`/`absoluteOffset`/`sizeBytes`/`dtype`/`elementBytes`. `MmapHandle.slice(at:count:)` available. `GTurboFormatV1.DType` has `.bf16`(1)/`.fp32`(3)/`.u32`(0).
- `ResidentEntry` now has explicit `init(... precomputed: Data? = nil)` and a `precomputed: Data?` field (custom init added because Swift's synthesized memberwise init did not expose the defaulted `Data?` property in this toolchain).

## Work State
### Completed
- Metal fixes: `qwen.metal` `log1p`→`log(1.0f+exp(x))`, `qwen_attn_qk_epilogue` `uint2`→`uint` 1D grid; `Qwen35Kernels.swift` 1D dispatch.
- `Tokenizing` protocol + shared `ChatMessage`/`ChatRole`/`ChatFunctionDefinition`/`ChatHistoricalToolCall` (ChatModel.swift); `GFTokenizer`/`Qwen3Tokenizer` conform.
- `Detokenizing` protocol + `QwenDetokenizer` + `Tokenizing.makeDetokenizer`; `runRawCompletion` uses `tokenizer.makeDetokenizer(...)`.
- Routed-expert streaming integrated into both Qwen prefill paths (`executePrefillGDNLayer`/`executePrefillFullAttnLayer`), both `async`.
- **Layout cap raised to 64MB** in `VerifiedInstallTool.swift` (`layoutMaxBytes`, `metadataMaxBytes`) and `PackedExpertsLayout.swift` (`defaultMaxBytes`) → `verify-install` passes: *"Verified 49 files (19540850389 bytes)"*.
- **Server `expecting` fix**: `ServerInference.swift` `Model.load(...)` passes `expecting: family.archConfig` (was defaulting to gemma4 2816).
- **Runner shared-expert bit fix**: `Qwen35ForwardRunner.swift` changed hardcoded `weightBits: 4` → `model.sharedExpertWeightBits` (manifest says 8).
- **Repack GDN fix (IMPLEMENTED + BUILDS)**: `RepackPlanner.swift` now has `forcedTargetDType(_:)` returning `.fp32` for `*.linear_attn.A_log` and `.bf16` for `*.linear_attn.(conv1d.weight|dt_bias|norm.weight)`. In `planResidentFile`, forced-path tensors are dequantized (U32 source → `dequantizeInt4ToFloating`, matching `dequant_int4.metal`) or converted (already-floating source → `convertFloating`) into `precomputed: Data` stored unquantized (no scales/biases). `ResidentWriter.write` emits `precomputed` bytes directly via `Posix.pwriteAll`. Full `swift build -c release` succeeds.

### Active
- User must re-pack Qwen to regenerate a correct `.gturbo` (the old package stored GDN tensors quantized). Then re-run the server to validate (may surface further Qwen runner runtime issues beyond the resident-schema check).

### Blocked
- (none) — fix is implemented; validation requires the user's source model + a full server load (heavy, ~19.5GB, needs Metal).

## Next Move
1. Instruct user to re-run `swift run -c release TurboFieldfareRepack --output scratch/qwen35.gturbo` (re-downloads/re-reads source; the tool now dequantizes GDN tensors correctly). If they want to keep the old file, suggest a new output path.
2. After repack, run `TurboFieldfareServer --model scratch/qwen35.gturbo --prefill-chunk-tokens 256` to validate (expect progress past the resident-schema error; watch for GDN/attention runtime issues).

## Relevant Files
- `Sources/TurboFieldfareRepack/Core/Planning/RepackPlanner.swift`: `ResidentEntry` (explicit init + `precomputed`), `planResidentFile` forced branch, `forcedTargetDType`, `dequantizeInt4ToFloating`, `convertFloating`, `readSourceBytes`, `bf16ToFloat`/`floatToBF16`/`floatValue`.
- `Sources/TurboFieldfareRepack/Core/Writing/ResidentWriter.swift`: `write` emits `precomputed` bytes.
- `Sources/TurboFieldfareRepack/Core/Format/IndexLoader.swift`: `quantSpec(forTensor:meta:)` (baseBits/overrides).
- `Sources/TurboFieldfare/Runtime/Inference/Model.swift`: `requireBF16` (604-620), `requireFP32` (819-833), `validateRuntimeSchema` GDN checks (~892-913).
- `Sources/TurboFieldfare/Metal/Quant/dequant_int4.metal`: canonical dequant formula.
- `Sources/TurboFieldfare/Infrastructure/ModelIO/PackedExpertsLayout.swift`: `defaultMaxBytes` (64MB).
- `Sources/TurboFieldfareRepack/Core/Verification/VerifiedInstallTool.swift`: `layoutMaxBytes`/`metadataMaxBytes` (64MB).
- `Sources/TurboFieldfareServer/Core/ServerInference.swift`: `expecting` fix.
- `Sources/TurboFieldfare/Runtime/Inference/Qwen35ForwardRunner.swift`: `sharedExpertWeightBits` fix.
