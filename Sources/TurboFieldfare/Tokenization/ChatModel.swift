import Foundation

/// Shared chat-message model used by every tokenizer backend (Gemma, Qwen, …).
/// Concrete tokenizers expose these as nested `typealias` so existing server
/// references (`GFTokenizer.Message`, `Qwen3Tokenizer.Message`, …) keep compiling.
public enum ChatRole: String, Sendable {
    case system, developer, user, assistant, tool
}

public struct ChatHistoricalToolCall: Sendable, Equatable {
    public let id: String
    public let name: String
    public let arguments: JSONValue

    public init(id: String, name: String, arguments: JSONValue) {
        self.id = id
        self.name = name
        self.arguments = arguments
    }
}

public struct ChatFunctionDefinition: Sendable, Equatable {
    public let name: String
    public let description: String
    public let parameters: JSONValue

    public init(name: String, description: String, parameters: JSONValue) {
        self.name = name
        self.description = description
        self.parameters = parameters
    }
}

public struct ChatMessage: Sendable, Equatable {
    public let role: ChatRole
    public let content: String?
    public let toolCalls: [ChatHistoricalToolCall]
    public let toolCallID: String?
    public let name: String?

    public init(role: ChatRole, content: String) {
        self.role = role
        self.content = content
        self.toolCalls = []
        self.toolCallID = nil
        self.name = nil
    }

    public init(role: ChatRole,
                content: String?,
                toolCalls: [ChatHistoricalToolCall] = [],
                toolCallID: String? = nil,
                name: String? = nil) {
        self.role = role
        self.content = content
        self.toolCalls = toolCalls
        self.toolCallID = toolCallID
        self.name = name
    }
}

/// Backend-agnostic tokenizer surface required by `ServerModelSession`.
/// Both `GFTokenizer` and `Qwen3Tokenizer` conform via extensions; the server
/// stores the active backend as `any Tokenizing`.
public protocol Tokenizing: Sendable {
    var toolCallStartID: Int32 { get }
    var toolCallEndID: Int32 { get }
    var toolResponseID: Int32 { get }
    var toolResponseEndID: Int32 { get }
    var endOfTurnID: Int32 { get }
    var channelStartID: Int32 { get }
    var channelEndID: Int32 { get }
    var stopTokenIDs: Set<Int32> { get }
    var structuralMarkerIDs: Set<Int32> { get }

    func applyChatTemplate(_ messages: [ChatMessage]) throws -> String
    func encode(_ text: String, addBOS: Bool) -> [Int32]
    func encodeTextContinuation(userContent: String) -> [Int32]
    func encodeToolChat(messages: [ChatMessage],
                        tools: [ChatFunctionDefinition]) throws -> [Int32]
    func encodeToolResultContinuation(cachedMessages: [ChatMessage],
                                      assistant: ChatMessage,
                                      incomingMessages: [ChatMessage],
                                      tools: [ChatFunctionDefinition]) throws -> [Int32]
    func decode(_ ids: [Int32], skipSpecialTokens: Bool) -> String
    func makeDetokenizer(barrierTokenIDs: Set<Int32>) -> any Detokenizing
}
