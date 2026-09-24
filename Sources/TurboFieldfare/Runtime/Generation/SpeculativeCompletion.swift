import Foundation
import Metal

/// Speculative-decode variant of `runRawCompletion`.
///
/// Uses a draft model on ANE to propose K candidate tokens per round, then
/// verifies them through the main Metal model. Accepted tokens are committed
/// in bulk; rejected tokens trigger KV rewind and correction.
///
/// The outer structure (prefill, stop handling, detokenizer, history) mirrors
/// `runRawCompletion` exactly so the two paths produce identical results for
/// the same prompt and config — the only difference is decode throughput.
public func runSpeculativeCompletion(
    runner: RealForwardRunner,
    speculativeSession: SpeculativeDecodeSession,
    tokenizer: GFTokenizer,
    promptIds: [Int32],
    multimodalInput: MultimodalPrefillInput? = nil,
    config: GenerationConfig,
    context: MetalContext,
    scratch: RawCompletionScratch,
    prefillConfig: PrefillRuntimeConfig = .defaultChunked,
    start: RawCompletionStart = .reset,
    shouldStop: () -> Bool = { false },
    onProgress: (RawDecodeProgress) -> Void
) async throws -> RawDecodeResult {
    try config.validate()
    guard !promptIds.isEmpty else {
        throw GeneratorError.emptyPrompt
    }

    let cachedPromptTokens = try start.cachedPromptTokens(
        promptCount: promptIds.count, producer: runner)

    if promptIds.count + config.maxNewTokens > runner.maxContext {
        throw GeneratorError.contextOverflow(prompt: promptIds.count,
                                              maxNew: config.maxNewTokens,
                                              maxContext: runner.maxContext)
    }

    // Synchronise the draft model's context with the prompt.
    speculativeSession.syncContext(tokens: promptIds, position: promptIds.count)

    // Prefill — identical to the non-speculative path.
    let prefill = try await runRawPrefill(
        producer: runner,
        promptIds: promptIds,
        multimodalInput: multimodalInput,
        prefillConfig: prefillConfig,
        start: start,
        outputMode: .logits,
        seedUse: .decode(isPureGreedy: config.isPureGreedy),
        historyReserve: promptIds.count + config.maxNewTokens,
        scratch: scratch,
        onProgress: onProgress)

    let computedPrefillTokens = prefill.computedPrefillTokens
    var history = prefill.history
    var position = prefill.position

    // Sync draft context after prefill: the draft model needs to see the
    // same token history the main model just prefilled.
    speculativeSession.syncContext(tokens: history, position: position)

    // Seed the speculative session's logits buffer with the main model's
    // prediction for the first decode position (position = prompt count).
    // This is the logits buffer the first verification round will check
    // against, so it must be populated before the decode loop starts.
    speculativeSession.seedLogits(from: scratch.logits)

    let decodeStart = Date()
    let prefillSeconds = prefill.prefillSeconds
    var detok = GFDetokenizer(tokenizer: tokenizer,
                              barrierTokenIDs: tokenizer.structuralMarkerIDs)
    var stopMatcher = StreamingStopMatcher(stops: config.stopStrings)
    var generated = 0
    var reason: StopReason = .maxTokens
    var uncommittedBoundaryTokenIDs: [Int32] = []
    var trailingInvisibleTokens = 0

    // Build the stop token ID set for the speculative session.
    var stopTokenIDs = Set(tokenizer.stopTokenIDs)
    stopTokenIDs.formUnion(config.extraStopTokens)

        // Speculative decode loop.
        while true {
            try Task.checkCancellation()

            // Check if we're near the context limit or speculative decoding
            // has been disabled by the fallback threshold.
            let remaining = runner.maxContext - position
            let useSpeculative = speculativeSession.isActive
                && remaining > speculativeSession.predictor.config.draftDepth + 1

        if useSpeculative {
            // --- Speculative round ---
            let roundResult = try await speculativeSession.runRound(
                runner: runner,
                config: config,
                context: context,
                stopTokenIDs: stopTokenIDs,
                position: position)

            // Emit accepted tokens.
            for tokenID in roundResult.acceptedTokens {
                generated += 1
                uncommittedBoundaryTokenIDs = [tokenID]

                let delta = detok.push(tokenID)
                let visible = stopMatcher.push(delta)
                onProgress(.token(index: generated - 1, id: tokenID, delta: visible))
                history.append(tokenID)
                trailingInvisibleTokens = visible.isEmpty ? trailingInvisibleTokens + 1 : 0
            }

            position += roundResult.acceptedTokens.count

            // Emit bonus token if present.
            if let bonusToken = roundResult.bonusToken {
                generated += 1
                uncommittedBoundaryTokenIDs = [bonusToken]

                let delta = detok.push(bonusToken)
                let visible = stopMatcher.push(delta)
                onProgress(.token(index: generated - 1, id: bonusToken, delta: visible))
                history.append(bonusToken)
                trailingInvisibleTokens = visible.isEmpty ? trailingInvisibleTokens + 1 : 0
                position += 1
            }

            // Check stop conditions.
            if roundResult.shouldStop {
                reason = roundResult.stopReason ?? .eos
                let tail = stopMatcher.push(detok.flush()) + stopMatcher.finish()
                if !tail.isEmpty { onProgress(.tail(tail)) }
                break
            }

            let hitMax = generated >= config.maxNewTokens
            let cancelled = shouldStop()
            if hitMax || cancelled {
                reason = hitMax ? .maxTokens : .cancelled
                let tail = stopMatcher.push(detok.flush()) + stopMatcher.finish()
                if !tail.isEmpty { onProgress(.tail(tail)) }
                break
            }
        } else {
            // --- Fallback: single-token decode (near context limit) ---
            let tokenID = try speculativeSampleOnce(
                scratch: scratch, context: context,
                history: history, config: config, position: generated)
            generated += 1
            uncommittedBoundaryTokenIDs = [tokenID]

            if stopTokenIDs.contains(tokenID) {
                reason = resolveStopReason(tokenID, tokenizer: tokenizer)
                let tail = stopMatcher.push(detok.flush()) + stopMatcher.finish()
                if !tail.isEmpty { onProgress(.tail(tail)) }
                break
            }

            let delta = detok.push(tokenID)
            let visible = stopMatcher.push(delta)
            onProgress(.token(index: generated - 1, id: tokenID, delta: visible))

            history.append(tokenID)
            trailingInvisibleTokens = visible.isEmpty ? trailingInvisibleTokens + 1 : 0
            try await runner.produceWithLogits(token: tokenID, position: position,
                                     into: scratch.logits)
            position += 1
            speculativeSession.syncContext(tokens: history, position: position)
            uncommittedBoundaryTokenIDs.removeAll(keepingCapacity: true)

            let hitMax = generated >= config.maxNewTokens
            let cancelled = shouldStop()
            if hitMax || cancelled {
                reason = hitMax ? .maxTokens : .cancelled
                let tail = stopMatcher.push(detok.flush()) + stopMatcher.finish()
                if !tail.isEmpty { onProgress(.tail(tail)) }
                break
            }
        }
    }

    return RawDecodeResult(prefillTokens: promptIds.count,
                           cachedPromptTokens: cachedPromptTokens,
                           computedPrefillTokens: computedPrefillTokens,
                           prefillSeconds: prefillSeconds,
                           newTokens: generated,
                           decodeSeconds: Date().timeIntervalSince(decodeStart),
                           reason: reason,
                           kvPosition: position,
                           kvBackedTokenIDs: history,
                           uncommittedBoundaryTokenIDs: uncommittedBoundaryTokenIDs,
                           withheldTrailingKVTokens: reason == .stopString
                               ? trailingInvisibleTokens : 0)
}

// MARK: - Helpers

/// Sample one token from the scratch logits buffer. Mirrors the private
/// `sampleOnce` in `RawCompletion.swift`.
private func speculativeSampleOnce(
    scratch: RawCompletionScratch,
    context: MetalContext,
    history: [Int32],
    config: GenerationConfig,
    position: Int
) throws -> Int32 {
    let cb = context.queue.makeCommandBuffer()!
    scratch.sampler.sample(commandBuffer: cb,
                           logits: scratch.logits,
                           probs: scratch.probs,
                           history: history,
                           config: config,
                           position: position,
                           outToken: scratch.outToken)
    cb.commit()
    cb.waitUntilCompleted()
    try checkCommandBufferError(cb)
    return Int32(bitPattern: scratch.outToken.contents().load(as: UInt32.self))
}

private func resolveStopReason(_ tokenID: Int32, tokenizer: GFTokenizer) -> StopReason {
    if tokenID == tokenizer.endOfTurnID { return .endOfTurn }
    if tokenID == tokenizer.toolResponseID { return .toolCalls }
    return .eos
}
