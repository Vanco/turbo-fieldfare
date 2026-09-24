# ANE Draft Model Speculative Decode Plan

## Overview

Add ANE (Apple Neural Engine) acceleration to TurboFieldfare using a small draft model for speculative decoding. A lightweight model runs on ANE via CoreML to predict K candidate tokens, then the main Metal-based 26B model verifies them in parallel. Expected speedup: 1.5-2x decode throughput.

## Architecture

```
┌─────────────────────────────────────────────────┐
│              Speculative Decode Loop             │
│                                                  │
│  ┌──────────────┐    ┌────────────────────────┐  │
│  │ Draft Model   │    │ Main Model (Metal)     │  │
│  │ (ANE/CoreML)  │───>│ 26B MoE, 30 layers     │  │
│  │ ~2-4B params  │    │ verify K tokens in     │  │
│  │ predict K     │    │ parallel via batched    │  │
│  │ tokens fast   │    │ forward pass            │  │
│  └──────────────┘    └────────────────────────┘  │
│         │                       │                │
│         └───────┬───────────────┘                │
│                 v                                 │
│          accept / reject tokens                   │
│          (acceptance rate ~70-85%)                │
└─────────────────────────────────────────────────┘
```

## Phase 1: CoreML Draft Model Integration (Foundation Layer) ✅ DONE

**Goal**: Create the CoreML draft engine module — load a small model on ANE, run single-token inference.

### Files Created

| File | Purpose |
|------|---------|
| `Sources/TurboFieldfare/DraftModel/DraftModelConfig.swift` | Configuration: model path, K value, temperature, ANE preferences |
| `Sources/TurboFieldfare/DraftModel/CoreMLDraftEngine.swift` | CoreML model loading, ANE placement, single-token inference |
| `Sources/TurboFieldfare/DraftModel/DraftPredictor.swift` | Autoregressive K-token prediction loop |

### Key Design Decisions

1. **CoreML as ANE gateway**: CoreML automatically routes compute to ANE when possible. `MLModelConfiguration.computeUnits = .all` enables ANE + GPU + CPU fallback.
2. **Draft model source**: Gemma 2B (or similar small model) converted via `coremltools` from MLX quantized format to `.mlpackage`.
3. **Module isolation**: All draft model code lives under `DraftModel/` — zero changes to existing Metal inference code in Phase 1.

### Implementation Details

#### DraftModelConfig.swift
- `struct DraftModelConfig: Sendable` — model URL, K (draft depth), temperature, compute units preference
- Validation: K in 1...10, model file exists
- Default production config for Gemma 2B drafting Gemma 26B

#### CoreMLDraftEngine.swift
- `final class CoreMLDraftEngine: @unchecked Sendable`
- `init(config: DraftModelConfig) throws` — loads `MLModel` with ANE-optimized configuration
- `func predictLogits(inputTokenIDs: [Int32]) throws -> [Float]` — runs one forward pass
- Uses `MLMultiArray` for input/output feature providers
- Configures `MLModelConfiguration`: computeUnits = .all, prefersANE = true

#### DraftPredictor.swift
- `final class DraftPredictor: @unchecked Sendable`
- `init(engine: CoreMLDraftEngine, config: DraftModelConfig)`
- `func predictKTokens(context: [Int32]) throws -> [Int32]` — autoregressive loop, K steps
- `func predictKTokensWithLogits(context: [Int32]) throws -> (tokens: [Int32], logits: [[Float]])` — for verification phase
- Greedy sampling by default (highest acceptance rate)

### Dependencies

- CoreML framework (system, no SPM dependency needed)
- Foundation
- No changes to `Package.swift` (CoreML is a system framework on macOS)

---

## Phase 2: Draft Token Prediction ✅ DONE

**Goal**: Wire draft predictor into the decode loop context.

### Files Created

| File | Purpose |
|------|---------|
| `Sources/TurboFieldfare/DraftModel/DraftKVContext.swift` | Draft model token context: sliding window, append, rewind, reset |
| `Sources/TurboFieldfare/DraftModel/SpeculativeDecodeSession.swift` | Speculative decode orchestration: draft → verify → accept/reject |

### Modified Files

| File | Change |
|------|--------|
| `Sources/TurboFieldfare/DraftModel/DraftPredictor.swift` | `config` access: `private` → `internal` |

---

## Phase 3: Parallel Verification (Core Algorithm) ✅ DONE

**Goal**: Main model verifies K draft tokens; integrate speculative path into decode loop.

### Files Created

| File | Purpose |
|------|---------|
| `Sources/TurboFieldfare/Runtime/Generation/SpeculativeCompletion.swift` | `runSpeculativeCompletion()` — speculative decode loop mirroring `runRawCompletion` |

### Modified Files

| File | Change |
|------|--------|
| `Sources/TurboFieldfare/DraftModel/SpeculativeDecodeSession.swift` | `predictor` access: `private` → `internal` |
| `Sources/TurboFieldfare/Runtime/Generation/MultimodalConversation.swift` | Added optional `speculativeSession` property; `generate()` routes to speculative path when available |

### Key Logic

```
1. Draft model produces K tokens: [t1, t2, ..., tK]
2. Main model runs K forward passes (batched or sequential)
3. For each position i: compare draft_token[i] vs main_token[i]
4. Accept all tokens before first mismatch
5. If all match, sample one bonus token from main model
6. KV cache truncated to accept point
```

---

## Phase 4: Acceptance Testing ✅ DONE

**Goal**: Token-level and distribution-level acceptance logic with statistics.

### Files Created

| File | Purpose |
|------|---------|
| `Sources/TurboFieldfare/DraftModel/AcceptanceTest.swift` | Acceptance strategies (tokenMatch, topK, probabilityThreshold), statistics tracking, softmax utilities |

### Modified Files

| File | Change |
|------|--------|
| `Sources/TurboFieldfare/DraftModel/DraftModelConfig.swift` | Added `acceptanceStrategy` property to config |
| `Sources/TurboFieldfare/DraftModel/SpeculativeDecodeSession.swift` | Uses `AcceptanceTest` for verification instead of simple `==` comparison; exposes acceptance statistics |

---

## Phase 5: KV Cache Synchronization ✅ DONE

**Goal**: Coordinate draft and main model KV cache state; snapshot/restore for rollback.

### Files Created

| File | Purpose |
|------|---------|
| `Sources/TurboFieldfare/DraftModel/KVCacheSync.swift` | Coordinates draft ↔ main KV state: batch commit, rejection rollback, snapshot/restore, consistency validation |

### Modified Files

| File | Change |
|------|--------|
| `Sources/TurboFieldfare/DraftModel/SpeculativeDecodeSession.swift` | Uses `KVCacheSync` for all context operations; snapshot at round start for rollback; `syncContext` and `reset` delegated to `KVCacheSync` |

---

## Phase 6: Configuration & Integration ✅ DONE

**Goal**: Expose speculative decoding in CLI and wire into decode path.

### Files Modified

| File | Change |
|------|--------|
| `Sources/TurboFieldfareCLI/Args.swift` | Added `--draft-model`, `--speculative-k`, `--acceptance-strategy` flags with parsing |
| `Sources/TurboFieldfareCLI/Run.swift` | Creates `SpeculativeDecodeSession` when `--draft-model` provided; routes to `runSpeculativeCompletion`; shows stats in footer |
| `Sources/TurboFieldfare/DraftModel/AcceptanceTest.swift` | `AcceptanceStrategy` now conforms to `CustomStringConvertible` |
| `Sources/TurboFieldfare/DraftModel/SpeculativeDecodeSession.swift` | Added `make()` public factory to avoid exposing internal types |

### Config Schema

```json
{
  "speculativeDecoding": {
    "enabled": true,
    "draftModelPath": "models/gemma-2b-it-coreml.mlpackage",
    "k": 5,
    "temperature": 0.0,
    "acceptanceThreshold": 1.0
  }
}
```

---

## Phase 7: Performance Tuning ✅ DONE

**Goal**: Adaptive K, fallback threshold, timing metrics.

### Modified Files

| File | Change |
|------|--------|
| `Sources/TurboFieldfare/DraftModel/DraftModelConfig.swift` | Added `enableAdaptiveK`, `fallbackThreshold` config fields |
| `Sources/TurboFieldfare/DraftModel/SpeculativeDecodeSession.swift` | Adaptive K via `predictor.adaptiveDraftDepth()`; fallback threshold disables speculative decode when acceptance too low; timing metrics (`draftModelSeconds`, `mainModelSeconds`) |
| `Sources/TurboFieldfare/Runtime/Generation/SpeculativeCompletion.swift` | Checks `speculativeSession.isActive` before using speculative path; falls back to single-token decode |
| `Sources/TurboFieldfareCLI/Args.swift` | Added `--no-adaptive-k` flag |
| `Sources/TurboFieldfareCLI/Run.swift` | Passes `enableAdaptiveK` to config; footer shows draft/main timing split |

---

## Expected Performance

| Device | Current Speed | Expected with Speculative Decode |
|--------|--------------|--------------------------------|
| M2 8GB | ~5.5 tok/s | ~8-12 tok/s (1.5-2x) |
| M5 Pro 24GB | ~33 tok/s | ~50-70 tok/s (1.5-2x) |

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| ANE memory pressure (two models) | Draft model <1B params; ANE has dedicated memory |
| CoreML conversion precision loss | FP16 quantization, minimal impact on draft model |
| Tokenizer incompatibility | Ensure both models use same tokenizer family |
| High reject rate | Adaptive K + fallback to non-speculative mode |
