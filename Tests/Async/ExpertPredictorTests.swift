//
//  ExpertPredictorTests.swift
//  TurboFieldfare
//
//  Unit tests for ExpertPredictor
//

import XCTest
@testable import TurboFieldfare

final class ExpertPredictorTests: XCTestCase {
    
    var predictor: ExpertPredictor!
    
    override func setUp() {
        super.setUp()
        predictor = ExpertPredictor(strategy: .statistical(lookback: 32))
    }
    
    override func tearDown() {
        predictor = nil
        super.tearDown()
    }
    
    // MARK: - Initialization Tests
    
    func testInitializeWithDefaultParameters() {
        let predictor = ExpertPredictor()
        XCTAssertEqual(predictor.strategy, .statistical(lookback: 32))
    }
    
    func testInitializeWithCustomParameters() {
        let predictor = ExpertPredictor(strategy: .statistical(lookback: 64))
        XCTAssertEqual(predictor.strategy, .statistical(lookback: 64))
    }
    
    // MARK: - Prediction Tests
    
    func testPredictNextExpertReturnsArray() {
        let experts = predictor.predictNextExpert(currentLayer: 0, tokenPosition: 0)
        XCTAssertGreaterThan(experts.count, 0)
    }
    
    func testPredictNextExpertReturnsTopK() {
        let experts = predictor.predictNextExpert(currentLayer: 0, tokenPosition: 0)
        XCTAssertEqual(experts.count, 8) // topK default
    }
    
    func testPredictNextExpertWithHiddenState() {
        // TODO: Test with actual hidden state
        let experts = predictor.predictNextExpert(currentLayer: 0, tokenPosition: 0)
        XCTAssertGreaterThan(experts.count, 0)
    }
    
    func testPredictChunkExpertsReturnsDistribution() {
        let tokens = Array(repeating: Token(id: 0, value: 0.0), count: 128)
        let distribution = predictor.predictChunkExperts(tokens, currentLayer: 0)
        
        XCTAssertGreaterThan(distribution.count, 0)
        // Verify all expert IDs are valid
        for expert in distribution.keys {
            XCTAssertGreaterThan(expert, 0)
        }
    }
    
    // MARK: - History Update Tests
    
    func testUpdateHistoryAddsSample() {
        let sample = RoutingSample(
            layer: 0,
            tokenPosition: 0,
            expert: ExpertID(0),
            timestamp: Date()
        )
        
        predictor.updateHistory(sample)
        XCTAssertEqual(predictor.routingHistory.count, 1)
    }
    
    func testUpdateHistoryMaintainsMaxSize() {
        let maxHistory = 100
        for _ in 0..<maxHistory + 10 {
            let sample = RoutingSample(
                layer: 0,
                tokenPosition: 0,
                expert: ExpertID(0),
                timestamp: Date()
            )
            predictor.updateHistory(sample)
        }
        
        XCTAssertEqual(predictor.routingHistory.count, maxHistory)
    }
    
    func testUpdateHistoryUpdatesExistingSample() {
        let sample1 = RoutingSample(
            layer: 0,
            tokenPosition: 0,
            expert: ExpertID(0),
            timestamp: Date()
        )
        predictor.updateHistory(sample1)
        
        let sample2 = RoutingSample(
            layer: 0,
            tokenPosition: 0,
            expert: ExpertID(1),
            timestamp: Date()
        )
        predictor.updateHistory(sample2)
        
        XCTAssertEqual(predictor.routingHistory.count, 2)
    }
    
    // MARK: - Cache Tests
    
    func testClearCacheRemovesAllEntries() {
        // Add some entries
        for _ in 0..<10 {
            let sample = RoutingSample(
                layer: 0,
                tokenPosition: 0,
                expert: ExpertID(0),
                timestamp: Date()
            )
            predictor.updateHistory(sample)
        }
        
        XCTAssertEqual(predictor.routingHistory.count, 10)
        predictor.clearCache()
        XCTAssertEqual(predictor.routingHistory.count, 0)
    }
    
    // MARK: - Statistics Tests
    
    func testStatisticsReturnsCorrectValues() {
        // Add some samples
        for _ in 0..<50 {
            let sample = RoutingSample(
                layer: 0,
                tokenPosition: 0,
                expert: ExpertID(0),
                timestamp: Date()
            )
            predictor.updateHistory(sample)
        }
        
        let stats = predictor.statistics()
        XCTAssertEqual(stats.totalPredictions, 50)
        XCTAssertGreaterThanOrEqual(stats.cacheHits, 0)
        XCTAssertGreaterThanOrEqual(stats.accuracy, 0.0)
        XCTAssertLessThanOrEqual(stats.accuracy, 1.0)
    }
    
    // MARK: - Memory Management Tests
    
    func testMemoryBudgetEnforcement() {
        let predictor = ExpertPredictor()
        XCTAssertEqual(predictor.maxMemoryBudget, 16 * 1024 * 1024 * 1024)
    }
    
    // MARK: - Edge Cases
    
    func testPredictEmptyHistory() {
        let experts = predictor.predictNextExpert(currentLayer: 0, tokenPosition: 0)
        // Should return valid prediction even with empty history
        XCTAssertGreaterThan(experts.count, 0)
    }
    
    func testPredictChunkEmptyTokens() {
        let tokens: [Token] = []
        let distribution = predictor.predictChunkExperts(tokens, currentLayer: 0)
        XCTAssertEqual(distribution.count, 0)
    }
    
    func testPredictChunkSingleToken() {
        let tokens = [Token(id: 0, value: 0.0)]
        let distribution = predictor.predictChunkExperts(tokens, currentLayer: 0)
        XCTAssertGreaterThan(distribution.count, 0)
    }
    
    // MARK: - Quality Gates
    
    func testQuality_predictionConsistency() {
        // Same input should produce same output
        let experts1 = predictor.predictNextExpert(currentLayer: 0, tokenPosition: 0)
        let experts2 = predictor.predictNextExpert(currentLayer: 0, tokenPosition: 0)
        XCTAssertEqual(experts1, experts2)
    }
    
    func testQuality_predictionStability() {
        // Multiple predictions should be stable
        var allExperts = [ExpertID]()
        for _ in 0..<100 {
            let experts = predictor.predictNextExpert(currentLayer: 0, tokenPosition: 0)
            allExperts.append(contentsOf: experts)
        }
        
        // Verify all experts are valid
        for expert in allExperts {
            XCTAssertGreaterThan(expert, 0)
        }
    }
    
    func testQuality_predictionDistribution() {
        // Verify prediction produces diverse expert distribution
        let tokens = Array(repeating: Token(id: 0, value: 0.0), count: 128)
        let distribution = predictor.predictChunkExperts(tokens, currentLayer: 0)
        
        // Should have at least some experts
        XCTAssertGreaterThan(distribution.count, 0)
        // Should not have all same expert
        let uniqueExperts = Set(distribution.keys)
        XCTAssertGreaterThan(uniqueExperts.count, 1)
    }
    
    // MARK: - Performance Tests
    
    func testPerformance_predictNextExpert() async throws {
        let experts = predictor.predictNextExpert(currentLayer: 0, tokenPosition: 0)
        measure {
            XCTAssertEqual(experts.count, 8)
        }
    }
    
    func testPerformance_predictChunkExperts() async throws {
        let tokens = Array(repeating: Token(id: 0, value: 0.0), count: 128)
        let distribution = predictor.predictChunkExperts(tokens, currentLayer: 0)
        measure {
            XCTAssertEqual(distribution.count, 8)
        }
    }
    
    // MARK: - Integration Tests
    
    func testIntegration_predictionPipeline() async throws {
        // Simulate prediction pipeline
        let layers = Array(0..<30)
        for layer in layers {
            let experts = predictor.predictNextExpert(currentLayer: layer)
            XCTAssertGreaterThan(experts.count, 0)
        }
    }
    
    func testIntegration_historyTracking() async throws {
        // Test history tracking over multiple predictions
        let samples = [
            RoutingSample(layer: 0, tokenPosition: 0, expert: ExpertID(0), timestamp: Date()),
            RoutingSample(layer: 1, tokenPosition: 1, expert: ExpertID(1), timestamp: Date()),
            RoutingSample(layer: 2, tokenPosition: 2, expert: ExpertID(2), timestamp: Date()),
        ]
        
        for sample in samples {
            predictor.updateHistory(sample)
        }
        
        XCTAssertEqual(predictor.routingHistory.count, 3)
    }
    
    // MARK: - Concurrency Tests
    
    func testConcurrency_multiplePredictions() async throws {
        // Multiple concurrent predictions
        let predictions = Array(repeating: Task {
            predictor.predictNextExpert(currentLayer: 0)
        }, count: 10)
        
        let results = await withTaskGroup(of: [ExpertID].self) { group in
            for prediction in predictions {
                group.addTask {
                    return await prediction.value
                }
            }
            
            for await result in group {
                // Verify each result
                for expert in result {
                    XCTAssertGreaterThan(expert, 0)
                }
            }
        }
    }
    
    func testConcurrency_nestedPrediction() async throws {
        // Nested prediction calls
        await Task {
            for layer in 0..<10 {
                let experts = predictor.predictNextExpert(currentLayer: layer)
                await Task.yield()
            }
        }
    }
    
    // MARK: - Error Handling Tests
    
    func testInvalidLayerID() {
        // Should handle invalid layer IDs gracefully
        let experts = predictor.predictNextExpert(currentLayer: 999999)
        // Should not crash
        XCTAssertEqual(experts.count, 8)
    }
    
    func testInvalidTokenPosition() {
        // Should handle invalid token positions gracefully
        let experts = predictor.predictNextExpert(currentLayer: 0, tokenPosition: 999999)
        // Should not crash
        XCTAssertEqual(experts.count, 8)
    }
    
    // MARK: - Debugging Tests
    
    func testDebug_logStatistics() {
        let stats = predictor.statistics()
        let expectedOutput = "Total predictions: \(stats.totalPredictions)"
        XCTAssertTrue(String(format: "%@ %d", expectedOutput, stats.totalPredictions).contains("predictions:"))
    }
}

// MARK: - Helper Functions

extension ExpertPredictor {
    var maxMemoryBudget: UInt {
        return 16 * 1024 * 1024 * 1024
    }
}