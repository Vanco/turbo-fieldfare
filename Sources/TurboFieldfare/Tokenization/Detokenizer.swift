import Foundation
import Tokenizers

/// Streaming detokenizer for generation loops. `GFTokenizer.decode` is a
/// push-loop over this type, so batch and streaming decode agree by
/// construction.
///
/// Emits each token's own contribution to the output as it arrives. Two
/// properties make that safe:
///
/// 1. The Gemma decoder sequence (`Replace`, `ByteFallback`, `Fuse`) is
///    position-independent, so the decode of a token stream is the
///    concatenation of its per-token fragments. `GemmaDecoding` reproduces that
///    sequence without HF's `clean_up_tokenization_spaces` pass, which is
///    the one stage that rewrites already-decoded text and is wrong for this
///    tokenizer anyway (see `GemmaDecoding`). ChatML/Qwen declare a byte-level
///    decoder instead, which `ByteLevelDecoding` reproduces; the two alphabets
///    are disjoint, so the dialect selects one and no token is ambiguous.
/// 2. BPE splits a multi-byte codepoint across several tokens — as `<0xXX>`
///    byte-fallback runs for Gemma, as raw byte runs for ChatML — so a run is
///    held until the token that closes it and commits as a whole, with the
///    reference decoder's semantics (see `ByteFallbackRun`, `ByteLevelRun`).
///    Skipped special tokens are filtered before the run logic — matching the
///    library, which drops special IDs before its decoder chain — so a run fuses
///    across them; in keep mode a special is an ordinary token and closes the
///    run.
///
/// `barrierTokenIDs` carves out an exception to the fuse rule for the
/// generation pipeline: the channel/tool markers structure assistant output,
/// and text held back across one would surface after the marker and be routed
/// under the wrong channel state (thought text leaking into the visible
/// answer). A barrier commits the run and returns its text as the marker's own
/// delta, which `StructuredAssistantDecoder.consume` routes under the channel
/// in effect before the marker switches it. Plain `decode` passes no barriers,
/// keeping full library parity.
///
/// Cost is O(1) per token and independent of how much has already been
/// generated. The previous implementation re-decoded the entire accumulated
/// token list on every push, which made a generation O(n²) — roughly 3.6·10⁹
/// dictionary lookups over the app's 64K-token budget — and compared the result
/// against the full emitted prefix to recover a delta.
struct GFDetokenizer {
    let tokenizer: any Tokenizer
    let skipSpecialTokens: Bool
    private let specialTokenIDs: Set<Int32>
    private let barrierTokenIDs: Set<Int32>
    /// ChatML/Qwen declare a byte-level decoder, Gemma a metaspace +
    /// byte-fallback one. The two alphabets are disjoint, so the dialect picks
    /// the path and no token is ambiguous.
    private let dialect: ChatDialect
    /// In-flight byte-fallback (Gemma) run.
    private var run = ByteFallbackRun()
    /// In-flight byte-level (ChatML) run.
    private var byteRun = ByteLevelRun()

    init(tokenizer: GFTokenizer,
         skipSpecialTokens: Bool = true,
         barrierTokenIDs: Set<Int32> = []) {
        self.tokenizer = tokenizer.tokenizer
        self.skipSpecialTokens = skipSpecialTokens
        self.specialTokenIDs = tokenizer.specialTokenIDs
        self.barrierTokenIDs = barrierTokenIDs
        self.dialect = tokenizer.dialect
    }

    /// Text contributed by `id`, ready to append to the stream.
    ///
    /// Returns `""` while a run is still open and valid; those bytes come out
    /// with the token that closes the run, at a barrier marker, or at
    /// `flush()`.
    mutating func push(_ id: Int32) -> String {
        // An unknown ID contributes nothing and leaves the run open, matching
        // the library, whose decode compactMap-drops unresolvable IDs.
        guard let token = tokenizer.convertIdToToken(Int(id)) else { return "" }
        if skipSpecialTokens, specialTokenIDs.contains(id) {
            return barrierTokenIDs.contains(id) ? commit() : ""
        }
        switch dialect {
        case .gemma:
            if let byte = GemmaDecoding.byteValue(token) { return run.push(byte) }
            return run.commit() + GemmaDecoding.fragment(token)
        case .chatml:
            // A character outside the byte alphabet means an added special
            // token, which is literal text rather than a byte run.
            guard let bytes = ByteLevelDecoding.bytes(for: token) else {
                return commit() + token
            }
            return commit() + byteRun.push(bytes)
        }
    }

    /// Remainder held back at a stop boundary.
    mutating func flush() -> String {
        commit()
    }

    private mutating func commit() -> String {
        switch dialect {
        case .gemma: return run.commit()
        case .chatml: return byteRun.commit()
        }
    }
}
