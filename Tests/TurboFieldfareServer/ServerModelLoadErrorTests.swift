import Testing
@testable import TurboFieldfare
@testable import TurboFieldfareServerCore

@Suite("Server model load errors")
struct ServerModelLoadErrorTests {

    @Test("Unsupported architecture names the family and the executable one")
    func unsupportedArchitectureMessage() {
        let error = ServerModelLoadError.unsupportedArchitecture(family: .qwen3_5_35B_A3B,
                                                                 tokenizerVerified: true)
        let description = String(describing: error)
        #expect(description.contains("qwen3_5_35B_A3B"))
        #expect(description.contains("loadable tokenizer"))
        #expect(description.contains("gemma4_26B_A4B"))
    }

    @Test("Missing sidecar asks for a reinstall")
    func missingSidecarMessage() {
        let error = ServerModelLoadError.missingTokenizerSidecar(family: .qwen3_5_35B_A3B)
        let description = String(describing: error)
        #expect(description.contains("qwen3_5_35B_A3B"))
        #expect(description.contains("reinstall"))
    }
}
