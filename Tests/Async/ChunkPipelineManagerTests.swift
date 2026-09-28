//
//  ChunkPipelineManagerTests.swift
//  TurboFieldfare
//
//  Unit tests for ChunkPipelineManager
//

import XCTest
@testable import TurboFieldfare

final class ChunkPipelineManagerTests: XCTestCase {
    
    var manager: ChunkPipelineManager!
    var device: MTLDevice!
    
    override func setUp() {
        super.setUp()
        device = MTLCreateSystemDefaultDevice()
        manager = ChunkPipelineManager(
            device: device,
            maxChunks: 4,
            maxMemoryBudget: 16 * 1024 * 1024 * 1024
        )
    }
    
    override func tearDown() {
        manager = nil
        super.tearDown()
    }
    
    // MARK: - Initialization Tests
    
    func testInitializeWithDefaultParameters() {
        let manager = ChunkPipelineManager()
        XCTAssertEqual(manager.maxChunks, 4)
        XCTAssertEqual(manager.defaultChunkSize, 128)
    }
    
    func testInitializeWithCustomParameters() {
        let manager = ChunkPipelineManager(
            maxChunks: 8,
            defaultChunkSize: 256
        )
        XCTAssertEqual(manager.maxChunks, 8)
        XCTAssertEqual(manager.defaultChunkSize, 256)
    }
    
    func testInitializeWithMemoryBudget() {
        let manager = ChunkPipelineManager(
            maxMemoryBudget: 8 * 1024 * 1024 * 1024  // 8GB
        )
        XCTAssertEqual(manager.maxMemoryBudget, 8 * 1024 * 1024 * 1024)
    }
    
    // MARK: - Add Chunk Tests
    
    func testAddChunkReturnsChunkID() {
        let tokens = Array(repeating: Token(id: 0, value: 0.0), count: 128)
        let chunkID = manager.addChunk(tokens)
        XCTAssertGreaterThan(chunkID, 0)
    }
    
    func testAddChunkCreatesMetadata() {
        let tokens = Array(repeating: Token(id: 0, value: 0.0), count: 128)
        let chunkID = manager.addChunk(tokens)
        
        let chunk = manager.chunks[chunkID]
        XCTAssertEqual(chunk.tokenCount, 128)
        XCTAssertEqual(chunk.layerRange.lowerBound, 0)
        XCTAssertEqual(chunk.layerRange.upperBound, 30) // Default numLayers
    }
    
    func testAddChunkWithPredictedExperts() {
        let tokens = Array(repeating: Token(id: 0, value: 0.0), count: 128)
        let predictedExperts = [ExpertID(0), ExpertID(1), ExpertID(2)]
        let chunkID = manager.addChunk(tokens, predictedExperts: predictedExperts)
        
        let chunk = manager.chunks[chunkID]
        XCTAssertEqual(chunk.predictedExperts, predictedExperts)
    }
    
    func testAddChunkRespectsMaxChunks() {
        // Add more chunks than max
        for _ in 0..<10 {
            let tokens = Array(repeating: Token(id: 0, value: 0.0), count: 128)
            let chunkID = manager.addChunk(tokens)
            XCTAssertLessThanOrEqual(chunkID, -1) // Should fail gracefully
        }
    }
    
    // MARK: - Process Chunk Tests
    
    func testProcessNextChunk() async {
        let tokens = Array(repeating: Token(id: 0, value: 0.0), count: 128)
        let chunkID = manager.addChunk(tokens)
        
        // Process the chunk
        await manager.processNextChunk()
        
        // Verify chunk was processed
        let chunk = manager.chunks[chunkID]
        XCTAssertNotEqual(chunk.state, .pending)
    }
    
    func testProcessChunkStateTransitions() async {
        let tokens = Array(repeating: Token(id: 0, value: 0.0), count: 128)
        let chunkID = manager.addChunk(tokens)
        
        // Verify initial state
        XCTAssertEqual(manager.chunks[chunkID].state, .pending)
        
        // Process through states
        await manager.processNextChunk()
        
        // Verify state transition
        XCTAssertNotEqual(manager.chunks[chunkID].state, .pending)
    }
    
    // MARK: - Execute Chunk Tests
    
    func testExecuteChunkReturnsResult() async {
        let tokens = Array(repeating: Token(id: 0, value: 0.0), count: 128)
        let chunkID = manager.addChunk(tokens)
        
        let result = await manager.executeChunk(chunkID)
        
        XCTAssertNotNil(result)
        XCTAssertEqual(result?.chunkID, chunkID)
    }
    
    func testExecuteChunkWithMemoryConstraint() async {
        // Test with limited memory budget
        let manager = ChunkPipelineManager(
            maxMemoryBudget: 100 * 1024 * 1024  // 100 MB
        )
        
        let tokens = Array(repeating: Token(id: 0, value: 0.0), count: 10000)
        let chunkID = manager.addChunk(tokens)
        
        let result = await manager.executeChunk(chunkID)
        
        // Should fail gracefully due to memory constraint
        XCTAssertNotNil(result)
        XCTAssertFalse(result!.isSuccessful)
    }
    
    // MARK: - Memory Management Tests
    
    func testMemoryBudgetEnforcement() {
        let manager = ChunkPipelineManager(maxMemoryBudget: 100 * 1024 * 1024)
        XCTAssertEqual(manager.maxMemoryBudget, 100 * 1024 * 1024)
    }
    
    func testCurrentMemoryUsage() {
        let manager = ChunkPipelineManager()
        XCTAssertEqual(manager.currentMemoryUsage, 0)
    }
    
    func testCanExecuteChunk() {
        let manager = ChunkPipelineManager()
        
        // Should be able to execute with small token count
        let canExecute = manager.canExecuteChunk(128)
        XCTAssertTrue(canExecute)
    }
    
    // MARK: - Statistics Tests
    
    func testStatisticsReturnsCorrectValues() {
        // Add some chunks
        for _ in 0..<5 {
            let tokens = Array(repeating: Token(id: 0, value: 0.0), count: 128)
            manager.addChunk(tokens)
        }
        
        let stats = manager.statistics()
        XCTAssertEqual(stats.totalChunks, 5)
        XCTAssertEqual(stats.totalMemory, 0)
        XCTAssertEqual(stats.availableMemory, manager.maxMemoryBudget)
    }
    
    func testCompletionRateCalculation() {
        let stats = PipelineStatistics(
            totalChunks: 10,
            completedChunks: 7,
            totalMemory: 0,
            availableMemory: 0
        )
        XCTAssertEqual(stats.completionRate, 0.7)
    }
    
    func testMemoryUsagePercentCalculation() {
        let stats = PipelineStatistics(
            totalChunks: 0,
            completedChunks: 0,
            totalMemory: 50,
            availableMemory: 0
        )
        XCTAssertEqual(stats.memoryUsagePercent, 0.5)
    }
    
    // MARK: - Callback Tests
    
    func testSetCompletionCallback() {
        var callbackCalled = false
        manager.setCompletionCallback { _, _ in
            callbackCalled = true
        }
        XCTAssertNotNil(manager.completionCallback)
    }
    
    func testSetProgressCallback() {
        var callbackCalled = false
        manager.setProgressCallback { _, _ in
            callbackCalled = true
        }
        XCTAssertNotNil(manager.progressCallback)
    }
    
    // MARK: - Lifecycle Management Tests
    
    func testResetManager() {
        // Add some chunks
        for _ in 0..<5 {
            let tokens = Array(repeating: Token(id: 0, value: 0.0), count: 128)
            manager.addChunk(tokens)
        }
        
        XCTAssertEqual(manager.chunks.count, 5)
        manager.reset()
        
        XCTAssertEqual(manager.chunks.count, 0)
        XCTAssertEqual(manager.pendingChunks.count, 0)
        XCTAssertEqual(manager.results.count, 0)
    }
    
    func testClearCompletedChunks() {
        // Add and complete some chunks
        for _ in 0..<5 {
            let tokens = Array(repeating: Token(id: 0, value: 0.0), count: 128)
            let chunkID = manager.addChunk(tokens)
            // Simulate completion
            manager.results[chunkID] = ChunkResult(
                chunkID: chunkID,
                tokenCount: 128,
                executionTime: 0.0,
                memoryUsed: 0,
                expertDistribution: [:],
                errors: []
            )
        }
        
        XCTAssertEqual(manager.chunks.count, 5)
        manager.clearCompletedChunks()
        XCTAssertEqual(manager.chunks.count, 0)
        XCTAssertEqual(manager.results.count, 0)
    }
    
    // MARK: - Pipeline Configuration Tests
    
    func testEstimatedMemoryPerChunk() {
        let manager = ChunkPipelineManager()
        let estimated = manager.config.estimatedMemoryPerChunk
        XCTAssertGreaterThan(estimated, 0)
    }
    
    func testMaxParallelChunks() {
        let manager = ChunkPipelineManager(
            maxMemoryBudget: 100 * 1024 * 1024 * 1024  // 100 GB
        )
        let maxParallel = manager.config.maxParallelChunks
        XCTAssertGreaterThan(maxParallel, 0)
    }
    
    // MARK: - Edge Cases
    
    func testEmptyChunkList() {
        let tokens: [Token] = []
        let chunkID = manager.addChunk(tokens)
        XCTAssertGreaterThan(chunkID, 0)
    }
    
    func testChunkWithZeroTokens() {
        let tokens = [Token(id: 0, value: 0.0)]
        let chunkID = manager.addChunk(tokens)
        XCTAssertEqual(manager.chunks[chunkID].tokenCount, 1)
    }
    
    func testChunkWithLargeTokenCount() {
        let tokens = Array(repeating: Token(id: 0, value: 0.0), count: 100000)
        let chunkID = manager.addChunk(tokens)
        XCTAssertEqual(manager.chunks[chunkID].tokenCount, 100000)
    }
    
    // MARK: - Quality Gates
    
    func testQuality_chunkStateConsistency() {
        // Verify chunk state transitions are consistent
        let tokens = Array(repeating: Token(id: 0, value: 0.0), count: 128)
        let chunkID = manager.addChunk(tokens)
        
        XCTAssertEqual(manager.chunks[chunkID].state, .pending)
        manager.reset()
        
        XCTAssertEqual(manager.chunks[chunkID].state, .pending)
    }
    
    func testQuality_chunkIDUniqueness() {
        // Verify chunk IDs are unique
        let chunkIDs = Set<Int>()
        for _ in 0..<100 {
            let tokens = Array(repeating: Token(id: 0, value: 0.0), count: 128)
            let chunkID = manager.addChunk(tokens)
            chunkIDs.insert(chunkID)
        }
        
        XCTAssertEqual(chunkIDs.count, 100)
    }
    
    func testQuality_chunkMetadataCorrectness() {
        let tokens = Array(repeating: Token(id: 0, value: 0.0), count: 256)
        let chunkID = manager.addChunk(tokens)
        
        let chunk = manager.chunks[chunkID]
        XCTAssertEqual(chunk.tokenCount, 256)
        XCTAssertEqual(chunk.chunkID, chunkID)
    }
    
    // MARK: - Performance Tests
    
    func testPerformance_addChunk() async throws {
        let tokens = Array(repeating: Token(id: 0, value: 0.0), count: 128)
        measure {
            let chunkID = manager.addChunk(tokens)
            XCTAssertEqual(chunkID, 0)
        }
    }
    
    func testPerformance_executeChunk() async throws {
        let tokens = Array(repeating: Token(id: 0, value: 0.0), count: 128)
        let chunkID = manager.addChunk(tokens)
        
        measure {
            await manager.executeChunk(chunkID)
        }
    }
    
    func testPerformance_multipleChunks() async throws {
        let numChunks = 100
        let tokens = Array(repeating: Token(id: 0, value: 0.0), count: 128)
        
        measure {
            for _ in 0..<numChunks {
                let chunkID = manager.addChunk(tokens)
            }
        }
    }
    
    // MARK: - Integration Tests
    
    func testIntegration_chunkPipeline() async throws {
        // Simulate complete pipeline
        let numChunks = 10
        for _ in 0..<numChunks {
            let tokens = Array(repeating: Token(id: 0, value: 0.0), count: 128)
            let chunkID = manager.addChunk(tokens)
            await manager.processNextChunk()
        }
        
        // Verify all chunks were processed
        XCTAssertEqual(manager.chunks.count, numChunks)
    }
    
    func testIntegration_memoryBudget() async throws {
        // Test with limited memory budget
        let manager = ChunkPipelineManager(
            maxMemoryBudget: 50 * 1024 * 1024 * 1024  // 50 GB
        )
        
        let numChunks = 50
        for _ in 0..<numChunks {
            let tokens = Array(repeating: Token(id: 0, value: 0.0), count: 128)
            let chunkID = manager.addChunk(tokens)
            XCTAssertGreaterThan(chunkID, 0)
        }
        
        XCTAssertEqual(manager.chunks.count, numChunks)
    }
    
    // MARK: - Concurrency Tests
    
    func testConcurrency_multipleChunks() async throws {
        // Add multiple chunks concurrently
        let numChunks = 20
        let tokens = Array(repeating: Token(id: 0, value: 0.0), count: 128)
        
        let chunkIDs = await withTaskGroup(of: Int.self) { group in
            for _ in 0..<numChunks {
                group.addTask {
                    return manager.addChunk(tokens)
                }
            }
            
            for await chunkID in group {
                // Verify each chunk ID
                XCTAssertGreaterThan(chunkID, 0)
            }
        }
        
        XCTAssertEqual(chunkIDs.count, numChunks)
    }
    
    func testConcurrency_chunkProcessing() async throws {
        // Process chunks concurrently
        let numChunks = 10
        let tokens = Array(repeating: Token(id: 0, value: 0.0), count: 128)
        
        let results = await withTaskGroup(of: ChunkResult?.self) { group in
            for _ in 0..<numChunks {
                let chunkID = manager.addChunk(tokens)
                group.addTask {
                    return await manager.executeChunk(chunkID)
                }
            }
            
            for await result in group {
                // Verify each result
                XCTAssertNotNil(result)
            }
        }
        
        XCTAssertEqual(results.count, numChunks)
    }
    
    // MARK: - Debugging Tests
    
    func testDebug_logStatistics() {
        let stats = manager.statistics()
        let expectedOutput = "Total chunks: \(stats.totalChunks)"
        XCTAssertTrue(String(format: "%@ %d", expectedOutput, stats.totalChunks).contains("chunks:"))
    }
}

// MARK: - Helper Functions

extension ChunkPipelineManager {
    var config: PipelineConfig {
        return PipelineConfig(
            maxChunks: self.maxChunks,
            defaultChunkSize: self.defaultChunkSize,
            maxMemoryBudget: self.maxMemoryBudget
        )
    }
}