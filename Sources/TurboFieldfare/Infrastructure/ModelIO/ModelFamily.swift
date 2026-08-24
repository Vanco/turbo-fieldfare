import Darwin
import Foundation
import TurboFieldfareFormat

public enum ModelFamilyError: Error, CustomStringConvertible, Equatable {
    case missingManifest(path: String)
    case unreadableManifest(path: String, detail: String)

    public var description: String {
        switch self {
        case .missingManifest(let path):
            return "model.gturbo directory at \(path) is missing manifest.json"
        case .unreadableManifest(let path, let detail):
            return "manifest.json at \(path) is not a readable GTURBO manifest: \(detail)"
        }
    }
}

/// Coarse model-family identity read from an installed `.gturbo` manifest.
///
/// Frontends use the family to pick the matching tokenizer and to reject
/// architectures the executable runtime does not implement yet with a precise
/// error instead of failing deep inside model loading. Detection is lenient:
/// only `magic` and `modelID` are consulted, so a manifest whose remaining
/// fields would fail full validation still reports its family.
public enum ModelFamily: String, Sendable, CaseIterable {
    case gemma4_26B_A4B
    case qwen3_5_35B_A3B

    /// Architecture baseline this family's manifest must match field-by-field.
    public var archConfig: ArchConfig {
        switch self {
        case .gemma4_26B_A4B: return .gemma4_26B_A4B
        case .qwen3_5_35B_A3B: return .qwen3_5_35B_A3B
        }
    }

    /// Whether the runtime implements this family's forward pass today.
    public var supportsRuntimeInference: Bool {
        self == .gemma4_26B_A4B
    }

    /// Read `manifest.json` far enough to identify the model family.
    public static func detect(modelDirectory: URL) throws -> ModelFamily {
        let directory = try GTurboModelDirectory(rootURL: modelDirectory)
        let fd: Int32
        do {
            fd = try directory.openFile("manifest.json")
        } catch {
            throw ModelFamilyError.missingManifest(path: directory.rootURL.path)
        }
        defer { close(fd) }
        let data: Data
        do {
            data = try directory.readMetadata(
                fileDescriptor: fd, relativePath: "manifest.json",
                maxBytes: ManifestReader.defaultMaxBytes)
        } catch {
            throw ModelFamilyError.unreadableManifest(path: directory.rootURL.path,
                                                      detail: "\(error)")
        }
        return try family(forManifestData: data, path: directory.rootURL.path)
    }

    static func family(forManifestData data: Data, path: String) throws -> ModelFamily {
        guard let wire = try? GTurboManifestCodec.decodeUnchecked(data),
              wire.magic == GTurboFormatV1.magic else {
            throw ModelFamilyError.unreadableManifest(path: path,
                                                      detail: "not a GTURBO v1 manifest")
        }
        return family(forModelID: wire.modelID)
    }

    /// Map a manifest `modelID` to its family. Unrecognized IDs resolve to the
    /// production Gemma 4 family so custom repacks keep today's behavior.
    static func family(forModelID modelID: String) -> ModelFamily {
        if modelID.lowercased().contains("qwen") {
            return .qwen3_5_35B_A3B
        }
        return .gemma4_26B_A4B
    }
}
