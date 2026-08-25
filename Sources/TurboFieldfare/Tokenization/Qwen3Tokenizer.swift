import Foundation
import Hub
import Tokenizers

// MARK: - Errors

public enum Qwen3TokenizerError: Error, CustomStringConvertible {
    case missingSpecialToken(String)
    case invalidChatTemplate(String)
    case missingTokenizerConfig
    case invalidTokenID(token: String, id: Int)

    public var description: String {
        switch self {
        case .missingSpecialToken(let t):
            return "Qwen3 tokenizer missing required special token: \(t)"
        case .invalidChatTemplate(let detail):
            return "Invalid Qwen3 chat messages: \(detail)"
        case .missingTokenizerConfig:
            return "Qwen3 tokenizer_config.json is missing or unreadable"
        case .invalidTokenID(let token, let id):
            return "Qwen3 tokenizer declares out-of-range ID \(id) for token \(token)"
        }
    }
}

// MARK: - Qwen3 Tokenizer

/// Tokenizer for Qwen3 models (including Qwen3-VL).
///
/// - Uses ChatML format: `<|im_start|>role\n...<|im_end|>`
/// - Special tokens are resolved from the tokenizer's `added_tokens`.
/// - Decoding uses the native `Tokenizers` pipeline (skipSpecialTokens = true).
    public struct Qwen3Tokenizer: @unchecked Sendable, Tokenizing {

    public static let defaultModelID = "Qwen/Qwen3-5B"

    // Essential special token IDs
    public let imStartID: Int32
    public let imEndID: Int32   // also EOS
    public let padID: Int32
    public let eosID: Int32

    /// Stop tokens: typically just `<|im_end|>`.
    public let stopTokenIDs: Set<Int32>

    /// Vocabulary size (fixed for Qwen3).
    public let vocabSize: Int

    @usableFromInline
    let tokenizer: any Tokenizer

    // MARK: - Loading

    public static func load(modelID: String = Self.defaultModelID) async throws -> Qwen3Tokenizer {
        try await Qwen3TokenizerLoadCoordinator.shared.load(.pretrained(modelID))
    }

    public static func load(from folder: URL) async throws -> Qwen3Tokenizer {
        try await Qwen3TokenizerLoadCoordinator.shared.load(.local(folder.standardizedFileURL.path))
    }

    /// Load from a model directory, optionally overriding via environment variable.
    public static func load(forModelDirectory modelDirectory: URL,
                            environment: [String: String] = ProcessInfo.processInfo.environment) async throws -> Qwen3Tokenizer {
        if let folder = tokenizerFolder(forModelDirectory: modelDirectory, environment: environment) {
            return try await load(from: folder)
        }
        return try await load()
    }

    public static func tokenizerFolder(forModelDirectory modelDirectory: URL,
                                       environment: [String: String] = ProcessInfo.processInfo.environment,
                                       fileManager: FileManager = .default) -> URL? {
        let sidecar = modelDirectory
            .standardizedFileURL
            .appendingPathComponent("tokenizer", isDirectory: true)
        if hasTokenizerJSON(in: sidecar, fileManager: fileManager) {
            return sidecar
        }

        guard let override = environment["Q3_TOKENIZER_DIR"], !override.isEmpty else {
            return nil
        }
        let overrideURL = URL(fileURLWithPath: override).standardizedFileURL
        return hasTokenizerJSON(in: overrideURL, fileManager: fileManager) ? overrideURL : nil
    }

    private static func hasTokenizerJSON(in folder: URL, fileManager: FileManager) -> Bool {
        fileManager.fileExists(atPath: folder.appendingPathComponent("tokenizer.json").path)
    }

    // MARK: - Internal Initialization

    /// Build the tokenizer from hub configuration.
    private static func make(from hub: LanguageModelConfigurationFromHub) async throws -> Qwen3Tokenizer {
        guard let tokenizerConfig = try await hub.tokenizerConfig else {
            throw Qwen3TokenizerError.missingTokenizerConfig
        }
        let tokenizerData = try await hub.tokenizerData
        let underlying = try AutoTokenizer.from(
            tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData)
        return try Qwen3Tokenizer(tokenizer: underlying, tokenizerData: tokenizerData)
    }

    static func loadUncached(pretrained modelID: String = Self.defaultModelID) async throws -> Qwen3Tokenizer {
        try await make(from: LanguageModelConfigurationFromHub(modelName: modelID))
    }

    static func loadUncached(from folder: URL) async throws -> Qwen3Tokenizer {
        try await make(from: LanguageModelConfigurationFromHub(modelFolder: folder))
    }

    // MARK: - Initializer

    public init(tokenizer: any Tokenizer, tokenizerData: Config) throws {
        self.tokenizer = tokenizer

        // Resolve required special tokens
        self.imStartID = try Self.requireTokenID(tokenizer, "<|im_start|>")
        self.imEndID   = try Self.requireTokenID(tokenizer, "<|im_end|>")
        self.eosID     = try Self.requireTokenID(tokenizer, "<|im_end|>") // same as imEnd

        // Pad token: Qwen resolves pad to "<|endoftext|>".
        self.padID = try Self.requireTokenID(tokenizer, "<|endoftext|>")

        self.stopTokenIDs = [eosID]
        self.vocabSize = 262_144 // Qwen3 base vocab size
    }

    // MARK: - Helper Functions

    /// Resolve a special token ID, rejecting `<unk>` substitution.
    private static func requireTokenID(_ tokenizer: any Tokenizer, _ token: String) throws -> Int32 {
        guard let id = tokenizer.convertTokenToId(token),
              tokenizer.convertIdToToken(id) == token else {
            throw Qwen3TokenizerError.missingSpecialToken(token)
        }
        return try int32ID(token, id)
    }

    private static func int32ID(_ token: String, _ id: Int) throws -> Int32 {
        guard let value = Int32(exactly: id) else {
            throw Qwen3TokenizerError.invalidTokenID(token: token, id: id)
        }
        return value
    }

    // MARK: - Encoding / Decoding

    /// Encode text to token IDs. No BOS is prepended by default (Qwen3 doesn't use BOS).
    public func encode(_ text: String, addBOS: Bool = false) -> [Int32] {
        // Use addSpecialTokens = false to avoid automatic insertion of BOS/EOS.
        return tokenizer.encode(text: text, addSpecialTokens: false).map(Int32.init)
    }

    /// Decode token IDs back to text, skipping special tokens by default.
    public func decode(_ ids: [Int32], skipSpecialTokens: Bool = true) -> String {
        let intIds = ids.map(Int.init)
        return tokenizer.decode(tokens: intIds, skipSpecialTokens: skipSpecialTokens)
    }

    // MARK: - Chat Template (ChatML)

    public typealias Role = ChatRole
    public typealias Message = ChatMessage

    /// Apply ChatML format: `<|im_start|>role\ncontent<|im_end|>` repeated, ending with `<|im_start|>assistant\n`.
    public func applyChatTemplate(_ messages: [Message]) throws -> String {
        try applyChatTemplate(messages, addGenerationPrompt: true)
    }

    private func applyChatTemplate(_ messages: [Message], addGenerationPrompt: Bool) throws -> String {
        var result = ""
        for message in messages {
            let role = message.role.rawValue
            let content = (message.content ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
            result += "<|im_start|>\(role)\n\(content)<|im_end|>\n"
        }
        if addGenerationPrompt {
            result += "<|im_start|>assistant\n"
        }
        return result
    }

    /// Encode the entire chat into token IDs using the ChatML template.
    public func encodeChat(_ messages: [Message], addGenerationPrompt: Bool = true) throws -> [Int32] {
        let text = try applyChatTemplate(messages, addGenerationPrompt: addGenerationPrompt)
        return encode(text, addBOS: false)
    }

    /// Convenience: encode a simple user prompt (without history).
    public func encodeUserPrompt(_ userContent: String) -> [Int32] {
        let messages = [Message(role: .user, content: userContent)]
        return (try? encodeChat(messages, addGenerationPrompt: true)) ?? []
    }

    // MARK: - Tokenizing protocol (Qwen)

    public var toolCallStartID: Int32 { imStartID }
    public var toolCallEndID: Int32 { imEndID }
    public var toolResponseID: Int32 { imEndID }
    public var toolResponseEndID: Int32 { imEndID }
    public var endOfTurnID: Int32 { imEndID }
    public var channelStartID: Int32 { -1 }
    public var channelEndID: Int32 { -1 }
    public var structuralMarkerIDs: Set<Int32> { [imEndID] }

    public func encodeTextContinuation(userContent: String) -> [Int32] {
        let content = userContent.trimmingCharacters(in: .whitespacesAndNewlines)
        let text = "<|im_end|>\n<|im_start|>user\n\(content)<|im_end|>\n<|im_start|>assistant\n"
        return encode(text, addBOS: false)
    }

    public func encodeToolChat(messages: [ChatMessage],
                               tools: [ChatFunctionDefinition]) throws -> [Int32] {
        throw Qwen3TokenizerError.invalidChatTemplate(
            "tool-call chat encoding is not yet implemented for Qwen in this runtime")
    }

    public func encodeToolResultContinuation(cachedMessages: [ChatMessage],
                                             assistant: ChatMessage,
                                             incomingMessages: [ChatMessage],
                                             tools: [ChatFunctionDefinition]) throws -> [Int32] {
        throw Qwen3TokenizerError.invalidChatTemplate(
            "tool-call continuation is not yet implemented for Qwen in this runtime")
    }

    public func makeDetokenizer(barrierTokenIDs: Set<Int32>) -> any Detokenizing {
        QwenDetokenizer(tokenizer: self, barrierTokenIDs: barrierTokenIDs)
    }
}

/// Streaming detokenizer for Qwen3: re-decodes the accumulated id buffer on each
/// push and emits the newly produced suffix. O(n^2) but correct; the Qwen
/// decoder is the upstream `Tokenizers` pipeline, so a true incremental path is
/// not available here.
struct QwenDetokenizer: Detokenizing {
    let tokenizer: Qwen3Tokenizer
    let skipSpecialTokens: Bool
    let barrierTokenIDs: Set<Int32>
    private var buffer: [Int32] = []
    private var lastText: String = ""

    init(tokenizer: Qwen3Tokenizer, skipSpecialTokens: Bool = true, barrierTokenIDs: Set<Int32> = []) {
        self.tokenizer = tokenizer
        self.skipSpecialTokens = skipSpecialTokens
        self.barrierTokenIDs = barrierTokenIDs
    }

    mutating func push(_ id: Int32) -> String {
        buffer.append(id)
        return emitDelta()
    }

    mutating func flush() -> String {
        return emitDelta()
    }

    private mutating func emitDelta() -> String {
        let full = tokenizer.decode(buffer, skipSpecialTokens: skipSpecialTokens)
        if full.hasPrefix(lastText) {
            let delta = String(full[lastText.endIndex...])
            lastText = full
            return delta
        }
        let delta = full
        lastText = full
        return delta
    }
}

// MARK: - Loading Coordinator

private enum Qwen3TokenizerLoadSource: Hashable {
    case pretrained(String)
    case local(String)
}

private actor Qwen3TokenizerLoadCoordinator {
    static let shared = Qwen3TokenizerLoadCoordinator()

    private var tasks: [Qwen3TokenizerLoadSource: Task<Qwen3Tokenizer, Error>] = [:]

    func load(_ source: Qwen3TokenizerLoadSource) async throws -> Qwen3Tokenizer {
        if let task = tasks[source] {
            return try await task.value
        }

        let task = Task.detached(priority: .userInitiated) { () throws -> Qwen3Tokenizer in
            switch source {
            case .pretrained(let modelID):
                return try await Qwen3Tokenizer.loadUncached(pretrained: modelID)
            case .local(let path):
                return try await Qwen3Tokenizer.loadUncached(from: URL(fileURLWithPath: path))
            }
        }
        tasks[source] = task

        do {
            return try await task.value
        } catch {
            tasks[source] = nil
            throw error
        }
    }
}
