import Foundation

/// The GPT-2 style byte-level decoder the ChatML/Qwen tokenizers declare
/// (`decoder: ByteLevel` in `tokenizer.json`).
///
/// Byte-level BPE does not put words in the vocabulary, it puts *byte
/// sequences*: the tokenizer first maps every one of the 256 byte values to a
/// printable Unicode character, and a token's string is the run of those
/// characters standing for its bytes. Decoding reverses the map and
/// interprets the result as UTF-8. That is why a raw ChatML token decodes to
/// `HelloĠthere!` under Gemma's SentencePiece rules — `Ġ` (U+0120) is the
/// stand-in for byte 0x20, and only the byte-level map turns it back into a
/// space.
///
/// `GemmaDecoding.fragment` cannot do this work: it substitutes `▁` (U+2581)
/// and has no mapping for `Ġ`, so a ChatML stream emitted the alphabet's
/// stand-in characters verbatim. The two alphabets are disjoint, so the two
/// paths never both apply to one token.
///
/// Byte-level tokens are *not* individually decodable — BPE may split a
/// multi-byte codepoint across two tokens — so the bytes stream through
/// `ByteLevelRun`, which emits a codepoint only once its last byte arrives.
enum ByteLevelDecoding {
    /// The 256 byte values, each mapped to the character that represents it in
    /// a byte-level vocabulary. Bytes that are already printable keep
    /// themselves; the rest move into U+0100… so the alphabet is valid text.
    ///
    /// Built once, densely indexed by byte, because this runs per byte on the
    /// decode hot path.
    private static let standIn: [Unicode.Scalar] = {
        var characters = [Unicode.Scalar?](repeating: nil, count: 256)
        // Printable ASCII, Latin-1 letters, and the general punctuation block
        // that HF's `bytes_to_unicode` keeps as-is.
        for byte in UInt8(0x21)...UInt8(0x7E) { characters[Int(byte)] = Unicode.Scalar(byte) }
        for byte in UInt8(0xA1)...UInt8(0xAC) { characters[Int(byte)] = Unicode.Scalar(byte) }
        for byte in UInt8(0xAE)...UInt8(0xFF) { characters[Int(byte)] = Unicode.Scalar(byte) }
        // Everything else continues past the end of Latin-1.
        var next: Unicode.Scalar = "\u{0100}"
        for index in characters.indices where characters[index] == nil {
            characters[index] = next
            next = Unicode.Scalar(next.value + 1)!
        }
        return characters.map { $0! }
    }()

    /// The byte a stand-in character stands for, or `nil` if the character is
    /// not part of the byte-level alphabet.
    ///
    /// Only the non-printable remainder needs a reverse lookup: the printable
    /// ranges are the identity, so their characters are their own byte.
    static func byte(for character: Character) -> UInt8? {
        guard let scalar = character.unicodeScalars.first,
              character.unicodeScalars.count == 1 else { return nil }
        if (0x21...0x7E).contains(scalar.value) || (0xA1...0xAC).contains(scalar.value)
            || (0xAE...0xFF).contains(scalar.value) {
            return UInt8(scalar.value)
        }
        guard (0x100...0x143).contains(scalar.value) else { return nil }
        return reverseStandIn[Int(scalar.value - 0x100)]
    }

    /// `byte(for:)` inverted for the U+0100…U+0143 block, computed once.
    private static let reverseStandIn: [UInt8] = {
        var table = [UInt8](repeating: 0, count: 0x44)
        for (offset, character) in standIn.enumerated() where character.value >= 0x100 {
            table[Int(character.value - 0x100)] = UInt8(offset)
        }
        return table
    }()

    /// The bytes `token` stands for, or `nil` if any of its characters is
    /// outside the alphabet — which means it is an added special token, not a
    /// byte-level word piece.
    static func bytes(for token: String) -> [UInt8]? {
        var bytes = [UInt8]()
        bytes.reserveCapacity(token.count)
        for character in token {
            guard let byte = byte(for: character) else { return nil }
            bytes.append(byte)
        }
        return bytes
    }
}

/// An in-flight run of byte-level bytes, held until each codepoint is
/// complete.
///
/// Unlike `ByteFallbackRun` this cannot poison: every byte value is
/// representable and BPE only splits *valid* UTF-8, so an incomplete tail is
/// always a lead byte waiting for its continuations. A byte that can never
/// complete one (a stray continuation, an overlong lead) is the tokenizer's
/// fault rather than the model's, so it decodes to one U+FFFD and the run keeps
/// going rather than poisoning everything after it.
struct ByteLevelRun {
    /// The incomplete codepoint at the head of the run, at most 3 bytes.
    private var bytes: [UInt8] = []
    /// Continuation bytes still owed, and the range the next one must fall in
    /// (RFC 3629 constrains the first continuation after E0/ED/F0/F4 leads).
    private var pendingContinuations = 0
    private var nextContinuation: ClosedRange<UInt8> = 0x80...0xBF

    /// Text the bytes so far complete, ready to append to the stream.
    mutating func push(_ incoming: [UInt8]) -> String {
        var out = ""
        for byte in incoming {
            if pendingContinuations == 0 {
                switch byte {
                case 0x00...0x7F:
                    out.unicodeScalars.append(Unicode.Scalar(byte))
                case 0xC2...0xDF:
                    begin(byte, continuations: 1)
                case 0xE0:
                    begin(byte, continuations: 2, next: 0xA0...0xBF)
                case 0xE1...0xEC, 0xEE, 0xEF:
                    begin(byte, continuations: 2)
                case 0xED:
                    begin(byte, continuations: 2, next: 0x80...0x9F)
                case 0xF0:
                    begin(byte, continuations: 3, next: 0x90...0xBF)
                case 0xF1...0xF3:
                    begin(byte, continuations: 3)
                case 0xF4:
                    begin(byte, continuations: 3, next: 0x80...0x8F)
                default:
                    // Stray continuation, overlong lead, or past U+10FFFF:
                    // unrepresentable, so it cannot become a codepoint.
                    out.unicodeScalars.append("\u{FFFD}")
                }
                continue
            }
            guard nextContinuation.contains(byte) else {
                // The lead can never complete. Flush what it owed as
                // replacement characters and re-offer this byte as a fresh
                // start, so one bad byte costs one U+FFFD.
                out += String(repeating: "\u{FFFD}", count: bytes.count + 1)
                bytes.removeAll(keepingCapacity: true)
                pendingContinuations = 0
                out += push([byte])
                return out
            }
            bytes.append(byte)
            nextContinuation = 0x80...0xBF
            pendingContinuations -= 1
            if pendingContinuations == 0 {
                out += String(bytes: bytes, encoding: .utf8)
                    ?? String(repeating: "\u{FFFD}", count: bytes.count)
                bytes.removeAll(keepingCapacity: true)
            }
        }
        return out
    }

    /// Remainder held back at a stop boundary: a trailing incomplete codepoint
    /// that no further token will complete.
    mutating func commit() -> String {
        defer {
            bytes.removeAll(keepingCapacity: true)
            pendingContinuations = 0
        }
        guard !bytes.isEmpty else { return "" }
        return String(bytes: bytes, encoding: .utf8)
            ?? String(repeating: "\u{FFFD}", count: bytes.count)
    }

    private mutating func begin(_ byte: UInt8, continuations: Int,
                                next: ClosedRange<UInt8> = 0x80...0xBF) {
        bytes.append(byte)
        pendingContinuations = continuations
        nextContinuation = next
    }
}
