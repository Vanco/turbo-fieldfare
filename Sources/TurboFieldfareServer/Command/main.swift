import Darwin
import Foundation
import TurboFieldfare
import TurboFieldfareServerCore

// Every request line goes to stderr, which is unbuffered, while the ready line
// below goes to stdout, which is fully buffered when it is not a terminal. A
// server started with its output redirected therefore showed an empty log for
// its whole life and printed "ready" only as it exited - exactly inverted from
// what an operator needs. Line buffering puts the line where it is useful.
setvbuf(stdout, nil, _IOLBF, 0)

let arguments: ServerArguments
let runtimeConfiguration: RuntimeConfiguration
do {
    arguments = try ServerArguments.parse(Array(CommandLine.arguments.dropFirst()))
    // Resolved here so an unusable flag combination exits with usage instead of
    // failing after the model has started loading.
    runtimeConfiguration = try arguments.resolvedRuntimeConfiguration()
} catch ServerArgumentError.help {
    print(ServerArguments.usage)
    exit(0)
} catch {
    FileHandle.standardError.write(Data("error: \(error)\n\n\(ServerArguments.usage)\n".utf8))
    exit(2)
}

do {
    let signals = ServerTerminationSignals()
    let modelURL = URL(fileURLWithPath: arguments.model).standardizedFileURL
    let backend = try await ServerModelSession.load(
        modelDirectory: modelURL,
        maxContext: arguments.maxContext,
        visionPackURL: arguments.visionPack.map {
            URL(fileURLWithPath: $0).standardizedFileURL
        },
        visionResidencyPolicy: arguments.visionResidency,
        promptCacheMode: arguments.promptCacheMode,
        runtimeConfiguration: runtimeConfiguration,
        arguments: arguments)
    // The family-derived default is what makes an installed Qwen model answer
    // as `qwen3.6-35b-a3b`; the override keeps `--model-id` authoritative.
    let modelID = arguments.modelIDOverride ?? backend.defaultModelID
    let server = TurboFieldfareHTTPServer(
        modelID: modelID,
        queueLimit: arguments.queueLimit,
        backend: backend,
        chatDialect: backend.chatDialect,
        visionCapability: backend.visionCapability)
    _ = try await server.start(port: arguments.port)
    // The slot count and prefill chunk size are the two knobs that decide
    // whether prefill is I/O bound, so the resolved values are printed rather
    // than the requested ones.
    let streaming: String
    let slots: Int
    switch arguments.expertStreamingMode {
    case .pread(let count):
        streaming = "pread"
        slots = count
    case .mmap(let count):
        streaming = "mmap"
        slots = count
    }
    print("TurboFieldfareServer ready on http://0.0.0.0:\(arguments.port) (reach it as http://<this-mac-lan-ip>:\(arguments.port); no auth, no TLS) model=\(modelID) context=\(arguments.maxContext) prompt_cache=\(arguments.promptCacheMode.rawValue) vision=\(backend.visionCapability) vision_residency=\(arguments.visionResidency.rawValue) expert_slots=\(slots) expert_streaming=\(streaming) prefill_chunk=\(arguments.prefillChunkTokens) thinking=\(arguments.enableThinking) dialect=\(backend.chatDialect.rawValue)")

    _ = await signals.wait()
    try await server.shutdown()
    await signals.cancel()
} catch {
    FileHandle.standardError.write(Data("error: \(error)\n".utf8))
    exit(1)
}
