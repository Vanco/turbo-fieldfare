import Foundation
import Metal
import TurboFieldfare

private struct MessageJSON: Decodable {
    let role: String
    let content: String
}

public struct RunResult: Equatable, Sendable {
    public let exitCode: Int32
    public init(exitCode: Int32) { self.exitCode = exitCode }
}

public func run(args: Args,
                stdout: FileHandle = .standardOutput,
                stderr: FileHandle = .standardError) async -> RunResult {
    do {
        let modelURL = URL(fileURLWithPath: args.model)
        let family = try ModelFamily.detect(modelDirectory: modelURL)

        let tokenizer: any Tokenizing
        switch family {
        case .gemma4_26B_A4B:
            let folder = GFTokenizer.tokenizerFolder(forModelDirectory: modelURL)!
            tokenizer = try await GFTokenizer.load(from: folder)
        case .qwen3_5_35B_A3B:
            let folder = Qwen3Tokenizer.tokenizerFolder(forModelDirectory: modelURL)!
            tokenizer = try await Qwen3Tokenizer.load(from: folder)
        }

        let promptIds: [Int32]
        if let tokenIDs = args.tokenIDs {
            promptIds = tokenIDs
                .split(separator: ",")
                .map { $0.trimmingCharacters(in: .whitespaces) }
                .compactMap { Int32($0) }
        } else if let rawPrompt = args.prompt {
            if family == .qwen3_5_35B_A3B {
                let messages = [GFTokenizer.Message(role: .user, content: rawPrompt)]
                let rendered = try tokenizer.applyChatTemplate(messages)
                promptIds = tokenizer.encode(rendered, addBOS: false)
            } else {
                promptIds = tokenizer.encode(rawPrompt, addBOS: true)
            }
        } else if let messagesFile = args.messagesFile {
            let data = try Data(contentsOf: URL(fileURLWithPath: messagesFile),
                                options: [.mappedIfSafe])
            let rows = try JSONDecoder().decode([MessageJSON].self, from: data)
            let messages = try rows.map { row -> GFTokenizer.Message in
                guard let role = GFTokenizer.Role(rawValue: row.role) else {
                    throw GFTokenizerError.invalidChatTemplate("unsupported role \(row.role)")
                }
                return GFTokenizer.Message(role: role, content: row.content)
            }
            let rendered = try tokenizer.applyChatTemplate(messages)
            promptIds = tokenizer.encode(rendered, addBOS: false)
        } else {
            return errored(stderr, "one of --prompt, --messages-file, or --token-ids is required", 2)
        }
        guard !promptIds.isEmpty else { return errored(stderr, "empty prompt", 2) }
        guard promptIds.count < args.maxContext else {
            return errored(
                stderr,
                "context overflow: prompt \(promptIds.count) reaches maxContext \(args.maxContext)",
                2)
        }
        let effectiveMaxNew = min(args.maxNew, args.maxContext - promptIds.count)
        let logitsOnly = args.logitsOut != nil
        let config = GenerationConfig(
            maxNewTokens: logitsOnly ? 1 : effectiveMaxNew,
            temperature: args.temperature,
            topK: args.topK,
            topP: args.topP,
            repetitionPenalty: args.repetitionPenalty,
            seed: args.seed,
            stopStrings: args.stops,
            extraStopTokens: [])
        let runtime = try args.resolvedRuntimeConfiguration(
            forceLogitsHead: !config.isPureGreedy)

        guard MTLCreateSystemDefaultDevice() != nil else {
            return errored(stderr, "no Metal device", 1)
        }
        let context = try MetalContext()
        let model = try Model.load(
            directoryURL: modelURL,
            device: context.device,
            expecting: family.archConfig,
            streamingMode: .pread(slotCount: runtime.expertCacheSlots),
            expertCachePolicy: runtime.modelExpertCachePolicy,
            integrityPolicy: .fullSha256)
        let runner: any LogitProducer
        switch family {
        case .gemma4_26B_A4B:
            runner = try RealForwardRunner(
                model: model,
                context: context,
                maxContext: args.maxContext,
                runtimeConfiguration: runtime)
        case .qwen3_5_35B_A3B:
            runner = try Qwen35ForwardRunner(
                model: model,
                context: context,
                maxContext: args.maxContext,
                runtimeConfiguration: runtime)
        }
        let scratch = try RawCompletionScratch(context: context,
                                               vocab: model.config.vocabSize)
        let logitsURL = args.logitsOut.map { URL(fileURLWithPath: $0) }
        let stats = try await runRawCompletion(
            producer: runner,
            tokenizer: tokenizer,
            promptIds: promptIds,
            config: config,
            context: context,
            scratch: scratch,
            prefillConfig: runtime.prefillConfig,
            onProgress: { progress in
                switch progress {
                case .prefill:
                    break
                case .token(_, _, let delta):
                    if !delta.isEmpty { stdout.write(Data(delta.utf8)) }
                case .tail(let tail):
                    stdout.write(Data(tail.utf8))
                }
            },
            onPrefillLogits: { buffer in
                guard let url = logitsURL else { return }
                let vocab = model.config.vocabSize
                let ptr = buffer.contents().bindMemory(to: Float16.self, capacity: vocab)
                var floats = [Float32](repeating: 0, count: vocab)
                for i in 0..<vocab { floats[i] = Float32(ptr[i]) }
                try? floats.withUnsafeBytes { try Data($0).write(to: url) }
            })

        if !args.quiet {
            let tokensPerSecond = stats.decodeSeconds > 0
                ? Double(stats.newTokens) / stats.decodeSeconds
                : 0
            let footer = "\n[stop=\(String(describing: stats.reason)) prefill=\(stats.prefillTokens)tok new=\(stats.newTokens)tok decode=\(String(format: "%.2f", stats.decodeSeconds))s tok/s=\(String(format: "%.3f", tokensPerSecond))]\n"
            stderr.write(Data(footer.utf8))
        }
        return RunResult(exitCode: 0)
    } catch is CancellationError {
        stdout.write(Data("\n".utf8))
        return RunResult(exitCode: 130)
    } catch {
        return errored(stderr, "\(error)", 1)
    }
}

private func errored(_ stderr: FileHandle, _ message: String, _ code: Int32) -> RunResult {
    stderr.write(Data("error: \(message)\n".utf8))
    return RunResult(exitCode: code)
}
