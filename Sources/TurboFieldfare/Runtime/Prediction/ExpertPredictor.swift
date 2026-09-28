//
//  ExpertPredictor.swift
//  TurboFieldfare
//
//  Expert predictor for MoE (Mixture of Experts) models.
//

import Foundation

/// Expert predictor that routes tokens to appropriate experts.
public final class ExpertPredictor {
    private let router: MoERouter
    private let topK: Int
    
    public init(router: MoERouter, topK: Int = 4) {
        self.router = router
        self.topK = topK
    }
    
    /// Predict which experts should handle a token.
    /// - Parameters:
    ///   - logits: Router logits for each expert
    ///   - token: Input token ID
    /// - Returns: List of selected expert IDs
    public func predictExperts(_ logits: [Double], token: Int) -> [Int] {
        let scores = router.computeScores(logits, forToken: token)
        return router.topK(scores, k: topK)
    }
    
    /// Get the output weights for selected experts.
    /// - Parameters:
    ///   - expertIDs: Selected expert IDs
    ///   - logits: Router logits
    /// - Returns: Weighted output logits
    public func computeExpertOutputs(expertIDs: [Int], logits: [Double]) -> [Double] {
        var outputLogits = [Double](repeating: 0.0, count: logits.count)
        
        for expertID in expertIDs {
            let weight = logits[expertID]
            if weight > 0 {
                for (i, value) in logits.enumerated() {
                    outputLogits[i] += weight * logits[expertID]
                }
            }
        }
        
        return outputLogits
    }
}

/// MoE router for expert selection.
public struct MoERouter {
    private let expertScores: [[Double]]
    private let numExperts: Int
    
    public init(expertScores: [[Double]], numExperts: Int) {
        self.expertScores = expertScores
        self.numExperts = numExperts
    }
    
    /// Compute routing scores for a token.
    /// - Parameters:
    ///   - logits: Router logits
    ///   - token: Input token ID
    /// - Returns: Scores for each expert
    public func computeScores(_ logits: [Double], forToken token: Int) -> [Double] {
        guard token < expertScores.count else {
            return Array(repeating: 0.0, count: numExperts)
        }
        return expertScores[token]
    }
    
    /// Get top K expert IDs by score.
    /// - Parameters:
    ///   - scores: Expert scores
    ///   - k: Number of top experts
    /// - Returns: Top K expert IDs
    public func topK(_ scores: [Double], k: Int) -> [Int] {
        var indexedScores: [(score: Double, index: Int)] = scores.enumerated().map { (score: $0.element, index: $0.offset) }
        indexedScores.sort { $0.score > $1.score }
        return indexedScores.prefix(k).map { $0.index }
    }
}
