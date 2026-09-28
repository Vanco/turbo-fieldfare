
# TurboFieldfare 异步优化分步实施方案
## 总体目标
将 prefill 阶段通过异步计算优化，在 16GB Mac mini 上实现：
- 短期（2-3 周）：层内异步，提升 15-25%
- 中期（3-4 周）：专家预取预测，提升 5-10%
- 长期（4-6 周）：Chunk 级流水线，提升 20-35%
---
## 阶段一：层内异步执行（Layer-Level Parallelism）
### 目标
将当前串行的层执行改为异步流水线，减少 CPU-GPU 同步点。
### 当前瓶颈
```
同步执行：
  GPU: 执行层 0 → 等待 → CPU: 读取路由 → 等待 → GPU: 执行层 1
  GPU: 执行层 1 → 等待 → CPU: 读取路由 → 等待 → GPU: 执行层 2
  ...
```
### 实施方案
#### 1.1 创建异步执行队列
文件: Sources/TurboFieldfare/Runtime/Inference/AsyncLayerExecutor.swift
```swift
/// 异步层执行器，管理 GPU 命令队列和 CPU 回调
class AsyncLayerExecutor {
    private var commandQueue: MTLCommandQueue
    private var layerQueue: [LayerExecution]
    private var callbacks: [LayerID: (Result) -> Void]
    
    func executeAsync(_ layers: [LayerID]) async {
        for layer in layers {
            // GPU: 准备命令
            let gpuCommand = prepareLayerCommand(layer)
            commandQueue.enqueue(gpuCommand)
            
            // CPU: 异步触发下一层准备
            Task {
                await prepareNextLayer(layer + 1)
            }
        }
    }
}
```
#### 1.2 修改 RealForwardRunner
文件: Sources/TurboFieldfare/Runtime/Inference/RealForwardRunner.swift
```swift
extension RealForwardRunner {
    func asyncRunChunk(_ chunk: [Token], async: Bool) async {
        guard async else {
            return runChunk(chunk) // 同步模式保持向后兼容
        }
        
        let executor = AsyncLayerExecutor()
        await executor.executeAsync(layers: 0..<numLayers)
    }
}
```
#### 1.3 内存预算分析
|资源	|当前	|阶段一需求	|16GB 剩余|
|----|----|----|----|
|模型权重	|1.35 GB	|+0.5 GB |异步缓冲区	✅|
|KV 缓存	|305 MiB	|+100 MiB |预取队列	✅|
|专家缓存	|1.50 GiB	|+500 MiB |预取区	✅|
|总需求	|~2.1 GB	|~2.6 GB	|✅|

#### 1.4 预期收益
```
基准（同步）：
  121 tokens: 9.34s
  527 tokens: 26.61s
  1017 tokens: 52.35s

预期（异步）：
  121 tokens: 7.00s  (-25%)
  527 tokens: 20.00s  (-25%)
  1017 tokens: 39.00s  (-25%)
```
#### 1.5 验证步骤
1. 创建测试用例 Tests/AsyncLayerExecutorTests.swift
2. 基准测试 docs/benchmark-prefill/async-baseline.json
3. A/B 对比：同步 vs 异步
4. 内存压力测试 memory_pressure -Q
---
## 阶段二：专家预取预测（Expert Prefetch Prediction）
### 目标
预测下一个 chunk 的专家分布，提前加载到缓存，减少磁盘 I/O。
### 预测模型选择
#### 选项 A：轻量级统计模型（推荐）
```python
# 基于历史路由频率的简单预测
def predict_next_experts(current_layer, token_positions):
    # 使用最近 32 个 token 的路由历史
    recent_history = router_history[-32:]
    expert_counts = Counter(recent_history)
    # 预测下一个 chunk 的专家分布
    return sample_from_distribution(expert_counts, top_k=8)
```
#### 选项 B：轻量级神经网络（备选）
```python
# 使用 128 个隐藏单元的 LSTM 预测路由
class RouterPredictor:
    def __init__(self, num_layers=30, hidden_size=128):
        self.model = LSTM(num_layers, hidden_size)
    
    def predict(self, hidden_states):
        # 输入：当前 hidden state
        # 输出：下一个 expert 分布
        return self.model(hidden_states)
```
### 实施方案
#### 2.1 创建预测模块
文件: Sources/TurboFieldfare/Runtime/Prediction/ExpertPredictor.swift
```swift
/// 专家分布预测器
enum ExpertPredictor {
    case statistical // 统计模型（默认）
    case neural // 神经网络
    
    /// 统计预测：基于历史路由频率
    static func statistical(history: [ExpertID], lookback: Int = 32) -> [ExpertID] {
        let recent = Array(history.suffix(lookback))
        let counts = Dictionary(grouping: recent) { $0 }.mapValues { $0.count }
        
        // 按频率排序
        let sorted = counts.sorted { $0.value > $1.value }
        
        // 采样 top-8 专家
        return sorted.prefix(8).map { $0.key }
    }
    
    /// 神经网络预测：基于 hidden state
    static func neural(hiddenState: Tensor<Float, 1>) async -> [ExpertID] {
        // TODO: 实现轻量级模型
        return await statistical(history: [])
    }
}
```
#### 2.2 集成到专家缓存
文件: Sources/TurboFieldfare/Infrastructure/Streaming/PreadExpertStreamer.swift
```swift
extension PreadExpertStreamer {
    /// 预取下一个 chunk 的专家
    func prefetchNextChunk(_ chunk: [Token], predictedExperts: [ExpertID]) async {
        for layer in predictedExperts {
            // 并行预取所有预测专家
            await withTaskGroup(of: Void) { group in
                for expert in layer {
                    group.addTask {
                        await self.fetchExpert(layer, expert)
                    }
                }
                await group.waitForAll()
            }
        }
    }
}
```
#### 2.3 内存预算分析
|资源	|当前	|阶段二需求	|16GB 剩余|
|----|----|----|----|
|预测模型	|0	|+10 MiB (统计) 或 +50 MiB (神经网络)	|✅|
|预取队列	|0	|+200 MiB (预测专家缓存)	|✅|
|总需求	|~2.1 GB	|~2.3-2.4 GB	|✅|

#### 2.4 预期收益
```
基准（无预取）：
  527 tokens: 26.61s
  1017 tokens: 52.35s

预期（统计预测）：
  527 tokens: 24.00s  (-10%)
  1017 tokens: 47.00s  (-9%)

预期（神经网络预测）：
  527 tokens: 22.00s  (-18%)
  1017 tokens: 43.00s  (-17%)
```
#### 2.5 验证步骤
1. 实现统计预测器 Tests/ExpertPredictorTests.swift
2. 评估预测准确率（hit rate）
3. 基准测试 docs/benchmark-prefill/prefetch-baseline.json
4. A/B 对比：无预测 vs 统计预测 vs 神经网络预测
----
## 阶段三：Chunk 级流水线（Chunk-Level Pipelining）
### 目标
同时处理多个 chunk，实现真正的并行化。
### 架构设计
```
Chunk 0: 执行中 (GPU)
Chunk 1: 预取专家 (CPU + GPU)
Chunk 2: 预测路由 (CPU)
Chunk 3: 等待执行 (CPU)

GPU 队列：
  [Chunk 0 Layer 0] → [Chunk 1 Layer 0] → [Chunk 2 Layer 0]
  [Chunk 0 Layer 1] → [Chunk 1 Layer 1] → [Chunk 2 Layer 1]
  ...
```
### 实施方案
#### 3.1 创建 Chunk 管理器
文件: Sources/TurboFieldfare/Runtime/Prefill/ChunkPipelineManager.swift
```swift
/// Chunk 级流水线管理器
class ChunkPipelineManager {
    private var chunks: [ChunkState]
    private var chunkQueue: MTLCommandQueue
    
    enum ChunkState {
        case pending // 等待处理
        case prefetching // 预取专家
        case predicting // 预测路由
        case executing // GPU 执行
        case completed // 完成
        
        var gpuCommand: MTLCommandBuffer?
        var predictedExperts: [ExpertID]?
    }
    
    /// 创建新的流水线
    init(maxChunks: Int = 4, chunkSize: Int = 128) {
        self.chunks = Array(repeating: .pending, count: maxChunks)
        self.chunkQueue = createCommandQueue()
    }
    
    /// 添加新 chunk 到流水线
    func addChunk(_ chunk: [Token]) async {
        // 找到可用的 chunk 槽位
        let slot = findAvailableSlot()
        
        switch chunks[slot].state {
        case .pending:
            // 直接开始执行
            await executeChunk(slot, chunk)
            
        case .prefetching:
            // 等待预取完成
            await chunks[slot].prefetchTask
            
        case .predicting:
            // 等待预测完成
            await chunks[slot].predictTask
            
        default:
            break
        }
    }
}
```
#### 3.2 扩展内存预算
|资源	|当前	|阶段三需求	|16GB 剩余|
|----|----|----|----|
|Chunk 缓冲区	|0	|+4 × 128 tokens = 64 KB	|✅|
|专家预取队列	|200 MiB	|+4 × 200 MiB = 800 MiB	|✅|
|GPU 队列	|0	|+2 × 1 GB = 2 GB	|⚠️|
|总需求	|~2.4 GB	|~3.0 GB	|✅|

注意：GPU 队列需要 2GB，这是主要内存压力点。
#### 3.3 预期收益
```
基准（单层）：
  1017 tokens: 52.35s

预期（4 chunk 并行）：
  1017 tokens: 35.00s  (-33%)

预期（8 chunk 并行）：
  1017 tokens: 28.00s  (-46%)
  内存需求：~3.5 GB
```
#### 3.4 验证步骤
1. 实现 Chunk 管理器 Tests/ChunkPipelineManagerTests.swift
2. 基准测试 docs/benchmark-prefill/pipeline-baseline.json
3. 不同并行度测试（2/4/8 chunks）
4. 内存压力测试
----
## 阶段四：MoE 批处理优化（MoE Batching Optimization）
### 目标
进一步优化 MoE 专家选择过程，减少分组开销。
### 当前问题

当前流程：
  1. 计算所有 token 的路由
  2. 排序 token-expert 对
  3. 分组相同专家
  4. 批量执行

### 优化方案
#### 4.1 流式 MoE 批处理
文件: Sources/TurboFieldfare/Kernels/Prefill/MoE/PrefillRoutedTileScheduler.swift
```swift
/// 流式 MoE 批处理调度器
class PrefillRoutedTileScheduler {
    /// 流式接收 token-expert 对，动态分组
    func streamBatch(tokens: [Token], experts: [ExpertID]) async -> [BatchResult] {
        // 使用环形缓冲区
        let ring = ExpertRingBuffer(capacity: 1024)
        
        // 流式处理
        for token in tokens {
            let expert = await predictExpert(for: token)
            ring.add(token, expert)
            
            // 当缓冲区满时，触发批处理
            if ring.isFull() {
                await processBatch(ring)
            }
        }
        
        // 处理剩余
        if !ring.isEmpty() {
            await processBatch(ring)
        }
    }
}
```
#### 4.2 预期收益
```
当前 MoE 时间：4.619 ms (PF-04)
优化后 MoE 时间：3.200 ms (-30%)

整体 prefill 提升：
  1017 tokens: 35.00s → 30.00s  (-14%)
```
----
## 阶段五：混合模式策略（Hybrid Mode）
### 目标
根据硬件和场景自动选择最优模式。
### 实现方案
文件: Sources/TurboFieldfare/Runtime/Configuration/AsyncMode.swift
```swift
/// 异步执行模式配置
enum AsyncMode {
    case disabled // 同步模式（向后兼容）
    case layerAsync // 层内异步
    case expertPrefetch // 专家预取
    case chunkPipeline // Chunk 级流水线
    case hybrid // 混合模式（推荐）
    
    /// 自动选择最优模式
    static func detectOptimalMode(
        availableMemory: UInt,
        chunkSize: Int,
        numChunks: Int
    ) -> AsyncMode {
        // 根据内存和配置选择
        guard availableMemory >= 3_000_000_000 else {
            return .layerAsync // 至少需要 3GB
        }
        
        guard chunkSize >= 128 else {
            return .layerAsync
        }
        
        guard numChunks <= 4 else {
            return .expertPrefetch // 4 chunks 是安全边界
        }
        
        return .hybrid // 最优模式
    }
}
```
### 预期整体收益
|模式	|121 tokens	|527 tokens	|1017 tokens	|6784 tokens|
|----|----|----|----|----|
|同步（当前）	|9.34s	|26.61s	|52.35s	|419.47s|
|层内异步	|7.00s	|20.00s	|39.00s	|314.00s|
|专家预取	|6.50s	|18.50s	|35.00s	|295.00s|
|Chunk 流水线	|5.50s	|16.00s	|30.00s	|250.00s|
|混合模式	|5.00s	|15.00s	|27.00s	|220.00s|

总体提升：

- 121 tokens: -46%
- 527 tokens: -43%
- 1017 tokens: -48%
- 6784 tokens: -47%

----
### 实施时间表
```
第 1 周：
  □ 阶段一设计评审
  □ 创建 AsyncLayerExecutor
  □ 编写单元测试

第 2 周：
  □ 阶段一集成到 RealForwardRunner
  □ 基准测试对比
  □ 修复 bug

第 3 周：
  □ 阶段二设计评审
  □ 实现 ExpertPredictor (统计)
  □ 集成到 PreadExpertStreamer

第 4 周：
  □ 阶段二基准测试
  □ 评估预测准确率
  □ 优化预测参数

第 5 周：
  □ 阶段三设计评审
  □ 实现 ChunkPipelineManager
  □ 内存预算分析

第 6 周：
  □ 阶段三基准测试
  □ 调试 GPU 队列
  □ 优化并行度

第 7 周：
  □ 阶段四设计评审
  □ 实现流式 MoE 批处理
  □ 集成到 PrefillRoutedTileScheduler

第 8 周：
  □ 阶段五设计评审
  □ 实现混合模式策略
  □ 完整基准测试

第 9 周：
  □ 文档更新
  □ 性能报告
  □ 准备发布
```
### 风险评估
|风险	|概率	|影响	|缓解措施|
|----|----|----|----|
|GPU 队列 OOM	|中	|高	|严格内存限制，动态调整 chunk 数|
|预测不准确	|高	|中	|降级到统计预测|
|CPU 瓶颈	|中	|中	|使用多核异步|
|调试困难	|高	|中	|添加详细日志和监控|
|向后兼容	|低	|高	|保持同步模式作为 fallback|
### 成功标准

- 所有单元测试通过
- 基准测试达到预期收益
- 内存使用不超过 16GB
- 质量指标（NLL, top-1）无下降
- 向后兼容同步模式
- 文档完整
