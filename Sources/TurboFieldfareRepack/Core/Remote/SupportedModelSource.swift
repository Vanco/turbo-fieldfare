import Foundation

/// 1. 定义模型配置协议（所有方法和属性均为 static）
public protocol ModelSourceConfiguration {
    static var displayName: String { get }
    static var repoID: String { get }
    static var revision: String { get }
    static var sourceIndexSHA256: String { get }
    static var approximateDownloadBytes: UInt64 { get }
    static var installedBytes: UInt64 { get }
    static var reserveBytes: UInt64 { get }

    static func installOptions(outputDirectory: URL,
                               overwrite: Bool,
                               token: String?,
                               resume: Bool) -> RemoteStreamingRepackOptions
}

/// 2. 为协议提供默认的 installOptions 实现（static）
extension ModelSourceConfiguration {
    public static func defaultInstallOptions(outputDirectory: URL,
                                             overwrite: Bool,
                                             token: String?,
                                             resume: Bool = false)
        -> RemoteStreamingRepackOptions {

        let mirrorURL = ProcessInfo.processInfo.environment["HF_ENDPOINT"]
            .flatMap { URL(string: $0) } ?? URL(string: "https://huggingface.co")!

        return RemoteStreamingRepackOptions(
            repoID: repoID,
            revision: revision,
            outputDir: outputDirectory.path,
            token: token,
            requireKnownSource: true,
            minFreeReserveBytes: reserveBytes,
            overwrite: overwrite,
            resume: resume,
            baseURL: mirrorURL)
    }
}

/// 3. 具体的模型实现：Gemma 4
public enum Gemma4Source: ModelSourceConfiguration {
    public static let displayName = "Gemma 4 26B-A4B IT 4-bit"
    public static let repoID = "mlx-community/gemma-4-26b-a4b-it-4bit"
    public static let revision = "0d77464eeb233a2da68ebf9d7dc4edaac7db956d"
    public static let sourceIndexSHA256 = "bf198c9f5ea6462addca1966e5dd669c407537a876e82cf06db9084c5c850b13"
    public static let approximateDownloadBytes: UInt64 = 14_620_479_420
    public static let installedBytes: UInt64 = 14_291_921_884
    public static let reserveBytes: UInt64 = 1_073_741_824

    public static func installOptions(outputDirectory: URL,
                                      overwrite: Bool,
                                      token: String?,
                                      resume: Bool = false)
        -> RemoteStreamingRepackOptions {
        return defaultInstallOptions(outputDirectory: outputDirectory,
                                     overwrite: overwrite,
                                     token: token,
                                     resume: resume)
    }
}

/// 4. 具体的模型实现：Qwen 3.5
public enum Qwen35Source: ModelSourceConfiguration {
    public static let displayName = "Qwen 3.5 35B-A3B 4-bit"
    public static let repoID = "mlx-community/Qwen3.5-35B-A3B-4bit"
    public static let revision = "c35ef17078b25552247b876b2b9cbfd1430bbfa9"
    public static let sourceIndexSHA256 = "56f02123353b7fe444b287a46a779a836cb939860502908da4a79e3b43931cdb"
    public static let approximateDownloadBytes: UInt64 = 20_391_405_152
    public static let installedBytes: UInt64 = 20_391_405_152
    public static let reserveBytes: UInt64 = 1_073_741_824

    public static func installOptions(outputDirectory: URL,
                                      overwrite: Bool,
                                      token: String?,
                                      resume: Bool = false)
        -> RemoteStreamingRepackOptions {
        return defaultInstallOptions(outputDirectory: outputDirectory,
                                     overwrite: overwrite,
                                     token: token,
                                     resume: resume)
    }
}

/// 5. 统一入口 (The Registry)
public enum SupportedModelSource {
    /// 当前选中的模型类型（存储类型本身）
    public nonisolated(unsafe) static var current: ModelSourceConfiguration.Type = Gemma4Source.self

    /// 所有支持的模型类型
    public nonisolated(unsafe) static let allSources: [ModelSourceConfiguration.Type] = [
        Gemma4Source.self,
        Qwen35Source.self
    ]
}
