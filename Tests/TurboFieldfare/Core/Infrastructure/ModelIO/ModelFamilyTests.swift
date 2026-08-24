import Foundation
import Testing
@testable import TurboFieldfare

@Suite("Model family detection")
struct ModelFamilyTests {

    // MARK: - modelID mapping

    @Test("Known repo IDs map to their families")
    func knownRepoIDsMapToFamilies() {
        #expect(ModelFamily.family(forModelID: "mlx-community/gemma-4-26b-a4b-it-4bit")
                == .gemma4_26B_A4B)
        #expect(ModelFamily.family(forModelID: "mlx-community/Qwen3.5-35B-A3B-4bit")
                == .qwen3_5_35B_A3B)
    }

    @Test("Matching is case-insensitive")
    func matchingIsCaseInsensitive() {
        #expect(ModelFamily.family(forModelID: "Org/QWEN3.5-Model") == .qwen3_5_35B_A3B)
        #expect(ModelFamily.family(forModelID: "ORG/GEMMA-4-IT") == .gemma4_26B_A4B)
    }

    @Test("Unrecognized model IDs default to the production family")
    func unrecognizedIDsDefaultToGemma() {
        #expect(ModelFamily.family(forModelID: "acme/other-model") == .gemma4_26B_A4B)
        #expect(ModelFamily.family(forModelID: "") == .gemma4_26B_A4B)
    }

    // MARK: - Capability

    @Test("Only Gemma 4 has an executable forward pass")
    func runtimeSupportFlags() {
        #expect(ModelFamily.gemma4_26B_A4B.supportsRuntimeInference)
        #expect(!ModelFamily.qwen3_5_35B_A3B.supportsRuntimeInference)
    }

    @Test("Qwen baseline mirrors the packed manifest")
    func qwenBaselineMirrorsPackedManifest() {
        let arch = ModelFamily.qwen3_5_35B_A3B.archConfig
        #expect(arch.hiddenSize == 2048)
        #expect(arch.intermediateSize == 512)
        #expect(arch.moeIntermediateSize == 512)
        #expect(arch.numHeads == 16)
        #expect(arch.numKVHeads == 2)
        #expect(arch.numFullKVHeads == 2)
        #expect(arch.headDim == 256)
        #expect(arch.fullHeadDim == 256)
        #expect(arch.vocabSize == 248_320)
        #expect(arch.slidingWindow == 1024)
        #expect(arch.finalLogitSoftcap == 30.0)
        #expect(arch.ropeTheta == 10_000.0)
        #expect(arch.fullRopeTheta == 1_000_000.0)
        #expect(arch.partialRotaryFactor == 0.25)
        #expect(arch.numLayers == 40)
        #expect(arch.numExperts == 256)
        #expect(arch.topKExperts == 8)
        #expect(!arch.tieWordEmbeddings)
        #expect(!arch.attentionKEqV)
        #expect(arch.fullAttentionLayerMask == [UInt8]([
            0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,
            0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,
            0, 0, 0, 1, 0, 0, 0, 1,
        ]))
        #expect(arch.hiddenActivation == "gelu_pytorch_tanh")
    }

    // MARK: - Filesystem detection

    @Test("Detection reads the installed manifest")
    func detectionReadsInstalledManifest() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let qwen = root.appendingPathComponent("qwen.gturbo", isDirectory: true)
        try makeModelDirectory(at: qwen, modelID: "mlx-community/Qwen3.5-35B-A3B-4bit")
        let gemma = root.appendingPathComponent("gemma.gturbo", isDirectory: true)
        try makeModelDirectory(at: gemma, modelID: "mlx-community/gemma-4-26b-a4b-it-4bit")

        #expect(try ModelFamily.detect(modelDirectory: qwen) == .qwen3_5_35B_A3B)
        #expect(try ModelFamily.detect(modelDirectory: gemma) == .gemma4_26B_A4B)
    }

    @Test("Missing manifest reports a missing manifest error")
    func missingManifestThrows() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let model = root.appendingPathComponent("model.gturbo", isDirectory: true)
        try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)

        #expect(throws: ModelFamilyError.missingManifest(path: model.standardizedFileURL.path)) {
            _ = try ModelFamily.detect(modelDirectory: model)
        }
    }

    @Test("Foreign manifest content reports an unreadable manifest")
    func foreignManifestThrows() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let model = root.appendingPathComponent("model.gturbo", isDirectory: true)
        try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
        try Data("{\"magic\":\"OTHER\"}".utf8)
            .write(to: model.appendingPathComponent("manifest.json"))

        #expect(throws: ModelFamilyError.unreadableManifest(
            path: model.standardizedFileURL.path,
            detail: "not a GTURBO v1 manifest")) {
            _ = try ModelFamily.detect(modelDirectory: model)
        }
    }

    // MARK: - Fixtures

    /// Minimal v1 manifest JSON. Detection decodes leniently, so only the
    /// non-optional wire fields are populated.
    private func makeModelDirectory(at url: URL, modelID: String) throws {
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        let manifest = """
        {
          "magic": "GTURBO",
          "versionMajor": 1,
          "versionMinor": 0,
          "flags": {},
          "modelID": "\(modelID)",
          "arch": {
            "hiddenSize": 2048, "ffnIntermediate": 512, "moeIntermediateSize": 512,
            "numHeads": 16, "numKVHeads": 2, "numFullKVHeads": 2,
            "headDim": 256, "fullHeadDim": 256, "vocabSize": 248320,
            "slidingWindow": 1024, "finalLogitSoftcap": 30,
            "ropeTheta": 10000, "fullRopeTheta": 1000000, "partialRotaryFactor": 0.25,
            "numLayers": 4, "numExperts": 256, "topKExperts": 8,
            "tieWordEmbeddings": false, "attentionKEqV": false,
            "hiddenActivation": "gelu_pytorch_tanh",
            "fullAttentionLayerMask": [0, 0, 0, 1]
          },
          "files": {},
          "expertsPerLayer": 256,
          "numLayers": 4,
          "expertStride": 16384
        }
        """
        try Data(manifest.utf8).write(to: url.appendingPathComponent("manifest.json"))
    }

    private func temporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("model-family-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }
}
