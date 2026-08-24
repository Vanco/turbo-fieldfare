import Foundation
import TurboFieldfareRepackCore

private let usage = """
    Usage:
      TurboFieldfareRepack --output <model.gturbo> [--overwrite] [--resume] [--model <name>]
      TurboFieldfareRepack --discard-partial --output <model.gturbo>
      TurboFieldfareRepack --verify-install --input-gturbo <model.gturbo>
      TurboFieldfareRepack --help

    Options:
      --model <name>        Choose model source: gemma4 (default) or qwen35
      --output <dir>        Output directory for the repacked model
      --overwrite           Overwrite existing output directory
      --resume              Resume a previously interrupted download
      --discard-partial     Discard saved partial download for a given output
      --verify-install      Verify integrity of an existing .gturbo installation
      --input-gturbo <dir>  Input .gturbo directory for verification

    The installer streams the model checkpoint from Hugging Face (or mirror set by
    HF_ENDPOINT) and repackages it without materializing the source checkpoint on disk.
    Set HF_TOKEN only if authentication is required.
    """

private struct Arguments {
    var output: String?
    var overwrite = false
    var resume = false
    var discardPartial = false
    var verifyInstall = false
    var inputGTurbo: String?
    var model: String?

    static func parse(_ values: [String]) throws -> Arguments {
        var parsed = Arguments()
        var index = 1
        while index < values.count {
            let flag = values[index]
            switch flag {
            case "--help":
                throw ParseError.help
            case "--overwrite":
                parsed.overwrite = true
                index += 1
            case "--resume":
                parsed.resume = true
                index += 1
            case "--discard-partial":
                parsed.discardPartial = true
                index += 1
            case "--verify-install":
                parsed.verifyInstall = true
                index += 1
            case "--output", "--input-gturbo", "--model":
                guard index + 1 < values.count else {
                    throw ParseError.missingValue(flag)
                }
                let value = values[index + 1]
                switch flag {
                case "--output": parsed.output = value
                case "--input-gturbo": parsed.inputGTurbo = value
                case "--model": parsed.model = value
                default: break
                }
                index += 2
            default:
                throw ParseError.unknown(flag)
            }
        }

        guard !(parsed.resume && parsed.discardPartial) else {
            throw ParseError.invalidMode("--resume and --discard-partial are mutually exclusive")
        }
        if parsed.discardPartial {
            guard parsed.output != nil else {
                throw ParseError.missingRequired("--output")
            }
            guard parsed.inputGTurbo == nil, !parsed.overwrite, !parsed.verifyInstall else {
                throw ParseError.invalidMode("--discard-partial only accepts --output")
            }
            return parsed
        }
        if parsed.verifyInstall {
            guard parsed.inputGTurbo != nil else {
                throw ParseError.missingRequired("--input-gturbo")
            }
            guard parsed.output == nil, !parsed.overwrite, !parsed.resume else {
                throw ParseError.invalidMode("verification accepts only --input-gturbo")
            }
        } else {
            guard parsed.output != nil else {
                throw ParseError.missingRequired("--output")
            }
            guard parsed.inputGTurbo == nil else {
                throw ParseError.invalidMode("--input-gturbo requires --verify-install")
            }
        }
        return parsed
    }
}

private enum ParseError: Error, CustomStringConvertible {
    case help
    case unknown(String)
    case missingValue(String)
    case missingRequired(String)
    case invalidMode(String)

    var description: String {
        switch self {
        case .help: return "help"
        case .unknown(let flag): return "unknown argument: \(flag)"
        case .missingValue(let flag): return "missing value for \(flag)"
        case .missingRequired(let flag): return "missing required argument: \(flag)"
        case .invalidMode(let message): return message
        }
    }
}

private func printError(_ message: String) {
    FileHandle.standardError.write(Data((message + "\n").utf8))
}

private func run(_ values: [String]) async -> Int32 {
    let arguments: Arguments
    do {
        arguments = try Arguments.parse(values)
    } catch ParseError.help {
        print(usage)
        return 0
    } catch {
        printError("error: \(error)\n\n\(usage)")
        return 2
    }

    if arguments.discardPartial, let output = arguments.output {
        do {
            try RemoteStreamingRepacker.discardPartial(outputDirectory: output)
            print("Discarded saved download for \(output)")
            return 0
        } catch {
            printError("discard failed: \(error)")
            return 1
        }
    }

    if arguments.verifyInstall, let input = arguments.inputGTurbo {
        do {
            let result = try VerifiedInstallTool.run(
                options: VerifyInstallOptions(inputGTurbo: input))
            print("Verified \(result.fileCount) files (\(result.bytesVerified) bytes)")
            print("Receipt: \(result.receiptPath)")
            return 0
        } catch {
            printError("verification failed: \(error)")
            return 1
        }
    }

    // 👇 如果用户指定了 --model，切换当前模型
    if let modelName = arguments.model {
        // 这里的逻辑：根据字符串匹配我们定义的 Source Type
        let matchedSource: ModelSourceConfiguration.Type?
        if modelName.lowercased() == "gemma4" {
            matchedSource = Gemma4Source.self
        } else if modelName.lowercased() == "qwen35" {
            matchedSource = Qwen35Source.self
        } else {
            matchedSource = nil
        }

        if let source = matchedSource {
            SupportedModelSource.current = source
        } else {
            printError("error: unknown model '\(modelName)'. Available: gemma4, qwen35")
            return 2
        }
    }

    // 使用当前选中的 source
    let activeSource = SupportedModelSource.current

    guard let output = arguments.output else { return 2 }
    let options = activeSource.installOptions(
        outputDirectory: URL(fileURLWithPath: output),
        overwrite: arguments.overwrite,
        token: ProcessInfo.processInfo.environment["HF_TOKEN"],
        resume: arguments.resume)
    do {
        let result = try await RemoteStreamingRepacker(options: options).run()
        print("Installed \(activeSource.displayName)")
        print("Source revision: \(result.resolvedCommit)")
        print("Model: \(result.outputDir)")
        return 0
    } catch {
        printError("install failed: \(error)")
        return 1
    }
}

exit(await run(CommandLine.arguments))
