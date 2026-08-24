import Foundation

/// Snapshot fingerprints pinned by the project. Adding a new entry means the
/// importer has been validated against a fresh upload of the source.
public enum SourceFingerprint {
    public static let knownFingerprints: [String: String] = {
        var dict: [String: String] = [:]
        for source in SupportedModelSource.allSources {
            dict[source.repoID] = source.sourceIndexSHA256
        }
        return dict
    }()

    public static func modelID(forIndexSha256 sha256Hex: String) -> String? {
        for source in SupportedModelSource.allSources {
            if source.sourceIndexSHA256 == sha256Hex {
                return source.repoID
            }
        }
        return nil
    }
}
