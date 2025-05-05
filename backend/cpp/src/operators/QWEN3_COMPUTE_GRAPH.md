# Qwen3 计算图与异构计算支持设计文档

## 1. 设计目标

本设计旨在为 Qwen3 模型提供更加灵活高效的推理支持，主要目标包括：

1. **计算图抽象**：使用计算图表示模型推理过程，实现更高效的执行和优化
2. **异构计算支持**：允许算子自由地在 CPU 和 GPU 之间调度执行
3. **自动内存管理**：在设备间自动进行内存复制和同步
4. **动态卸载策略**：根据内存压力和计算需求，动态决定哪些算子卸载到 CPU
5. **流水线并行**：支持 CPU 和 GPU 之间的流水线并行执行

## 2. 架构设计

### 2.1 核心组件

#### 2.1.1 计算图（ComputeGraph）

计算图是由节点（算子）和边（张量）组成的有向无环图（DAG），表示整个模型的计算流程。

```cpp
class ComputeGraph {
public:
    // 添加节点到计算图
    NodeHandle addNode(std::shared_ptr<OperatorNode> node);
    
    // 设置节点间的数据依赖关系
    void addEdge(NodeHandle from, NodeHandle to, int output_idx = 0, int input_idx = 0);
    
    // 执行整个计算图
    void execute(cudaStream_t stream = nullptr);
    
    // 优化计算图（融合算子、内存规划等）
    void optimize();
    
    // 设置内存预算，用于动态卸载决策
    void setMemoryBudget(size_t gpu_memory_budget);
    
private:
    std::vector<std::shared_ptr<OperatorNode>> nodes_;
    std::unordered_map<NodeHandle, std::vector<EdgeInfo>> edges_;
    std::unordered_map<NodeHandle, std::vector<EdgeInfo>> reverse_edges_;
    
    // 拓扑排序结果，用于执行
    std::vector<NodeHandle> execution_order_;
    
    // 内存规划
    MemoryPlanner memory_planner_;
};
```

#### 2.1.2 算子节点（OperatorNode）

表示计算图中的一个计算节点，封装了具体的算子实现和执行逻辑。

```cpp
class OperatorNode {
public:
    // 执行算子
    virtual void execute(cudaStream_t stream = nullptr) = 0;
    
    // 获取输入/输出张量
    virtual Tensor<T>* getInput(int idx = 0) = 0;
    virtual Tensor<T>* getOutput(int idx = 0) = 0;
    
    // 设置首选执行设备
    void setPreferredDevice(Device device);
    
    // 获取算子的内存需求（用于内存规划）
    virtual MemoryRequirement getMemoryRequirement() = 0;
    
    // 获取算子的计算复杂度（用于调度决策）
    virtual float getComputeComplexity() = 0;
    
protected:
    std::vector<Tensor<T>*> inputs_;
    std::vector<Tensor<T>*> outputs_;
    Device preferred_device_ = Device::CUDA;
    bool allow_offload_ = true;  // 是否允许卸载到CPU
};
```

#### 2.1.3 内存规划器（MemoryPlanner）

负责计算图的内存分配和管理，包括张量生命周期分析和内存复用。

```cpp
class MemoryPlanner {
public:
    // 分析计算图，确定张量生命周期
    void analyze(const ComputeGraph& graph);
    
    // 规划内存分配，尽量复用内存
    void plan(size_t gpu_memory_budget);
    
    // 获取张量的实际存储位置
    TensorStorage* getTensorStorage(TensorHandle tensor);
    
private:
    std::unordered_map<TensorHandle, LifetimeInfo> tensor_lifetimes_;
    std::unordered_map<TensorHandle, TensorStorage*> tensor_storage_;
    
    // 内存池管理
    std::unique_ptr<GPUMemoryPool> gpu_pool_;
    std::unique_ptr<CPUMemoryPool> cpu_pool_;
};
```

#### 2.1.4 设备间同步管理器（DeviceSyncManager）

管理不同设备间的数据传输和同步。

```cpp
class DeviceSyncManager {
public:
    // 确保张量在指定设备上可用
    void ensureAvailable(Tensor<T>* tensor, Device target_device, cudaStream_t stream = nullptr);
    
    // 异步数据传输
    void transferAsync(Tensor<T>* src, Tensor<T>* dst, cudaStream_t stream = nullptr);
    
    // 注册事件依赖
    void addDependency(cudaEvent_t event, cudaStream_t dependent_stream);
    
private:
    // 缓存已传输的张量，避免重复传输
    std::unordered_map<Tensor<T>*, DeviceStatus> tensor_status_;
    
    // 流和事件管理
    StreamPool stream_pool_;
};
```

### 2.2 异构计算支持

#### 2.2.1 设备选择策略

1. **静态策略**：根据算子类型和输入大小预先确定最佳设备
2. **动态策略**：根据运行时系统负载和内存压力动态调整
3. **混合策略**：结合静态分析和动态调整

```cpp
class DeviceSelector {
public:
    // 为算子选择最佳执行设备
    Device selectDevice(OperatorNode* node, SystemStatus& status);
    
    // 更新性能统计信息
    void updateStats(OperatorNode* node, Device device, float execution_time);
    
private:
    // 算子在不同设备上的性能统计
    std::unordered_map<OperatorType, DevicePerformanceStats> performance_stats_;
    
    // 负载均衡器
    LoadBalancer load_balancer_;
};
```

#### 2.2.2 内存管理与数据传输

1. **延迟传输**：仅在必要时才在设备间传输数据
2. **预取机制**：预测即将需要的数据并提前传输
3. **保留策略**：根据使用频率决定是否在设备间保留数据副本

```cpp
class MemoryManager {
public:
    // 分配设备内存
    void* allocate(size_t size, Device device);
    
    // 释放设备内存
    void free(void* ptr, Device device);
    
    // 在设备间传输数据
    void transfer(void* src, Device src_device, void* dst, Device dst_device, size_t size, cudaStream_t stream = nullptr);
    
    // 设置内存预算
    void setMemoryBudget(Device device, size_t budget);
    
private:
    // 设备内存池
    std::unordered_map<Device, std::unique_ptr<MemoryPool>> memory_pools_;
    
    // 内存压力监控
    MemoryPressureMonitor pressure_monitor_;
};
```

## 3. 计算图构建与执行

### 3.1 图构建接口

提供直观的接口用于构建计算图：

```cpp
// 示例：构建一个简单的MLP前向计算图
ComputeGraph buildMLPGraph(Tensor<T>* input, const std::vector<Tensor<T>*>& weights, const std::vector<Tensor<T>*>& biases) {
    ComputeGraph graph;
    
    // 添加第一层线性变换
    auto matmul1 = graph.addNode(std::make_shared<MatMulNode>(input, weights[0]));
    auto bias1 = graph.addNode(std::make_shared<AddNode>(nullptr, biases[0]));
    graph.addEdge(matmul1, bias1);
    
    // 添加激活函数
    auto gelu = graph.addNode(std::make_shared<GELUNode>());
    graph.addEdge(bias1, gelu);
    
    // 添加第二层线性变换
    auto matmul2 = graph.addNode(std::make_shared<MatMulNode>(nullptr, weights[1]));
    graph.addEdge(gelu, matmul2);
    auto bias2 = graph.addNode(std::make_shared<AddNode>(nullptr, biases[1]));
    graph.addEdge(matmul2, bias2);
    
    return graph;
}
```

### 3.2 图执行流程

1. **初始化**：准备输入张量和参数
2. **拓扑排序**：确定节点执行顺序
3. **设备分配**：为每个节点选择执行设备
4. **内存规划**：分析张量生命周期，规划内存分配
5. **执行**：按拓扑顺序执行节点，处理设备间数据传输
6. **清理**：释放临时内存

```cpp
void ComputeGraph::execute(cudaStream_t main_stream) {
    // 如果尚未进行拓扑排序，先排序
    if (execution_order_.empty()) {
        topologicalSort();
    }
    
    // 创建执行上下文
    ExecutionContext context(main_stream);
    
    // 按拓扑顺序执行节点
    for (auto node_handle : execution_order_) {
        auto node = nodes_[node_handle];
        
        // 确保输入张量在正确的设备上
        for (int i = 0; i < node->getInputCount(); ++i) {
            auto input = node->getInput(i);
            if (input) {
                context.sync_manager.ensureAvailable(input, node->getExecutionDevice(), context.getStream(node->getExecutionDevice()));
            }
        }
        
        // 执行节点
        node->execute(context.getStream(node->getExecutionDevice()));
        
        // 记录完成事件
        if (node->getExecutionDevice() == Device::CUDA) {
            cudaEvent_t event;
            cudaEventCreate(&event);
            cudaEventRecord(event, context.getStream(Device::CUDA));
            context.addEvent(node_handle, event);
        }
    }
    
    // 同步主流
    if (main_stream) {
        for (auto& [node_handle, event] : context.events) {
            cudaStreamWaitEvent(main_stream, event, 0);
        }
    } else {
        // 等待所有操作完成
        cudaDeviceSynchronize();
    }
}
```

## 4. 动态卸载与流水线并行

### 4.1 动态卸载策略

根据以下因素决定是否将算子卸载到CPU：

1. **内存压力**：当GPU内存接近预算上限时
2. **计算特性**：计算密集型算子倾向于在GPU上执行，内存密集型算子倾向于在CPU上执行
3. **数据依赖**：考虑数据传输开销
4. **历史性能**：基于历史执行时间的统计

```cpp
class OffloadingStrategy {
public:
    // 决定是否应该卸载算子到CPU
    bool shouldOffload(OperatorNode* node, SystemStatus& status);
    
    // 更新卸载决策的统计信息
    void updateStats(OperatorNode* node, bool was_offloaded, float execution_time);
    
private:
    // 算子卸载历史
    std::unordered_map<OperatorType, OffloadStats> offload_stats_;
    
    // 内存压力阈值
    float memory_pressure_threshold_ = 0.8f;
    
    // 性能比例阈值（CPU时间/GPU时间）
    float performance_ratio_threshold_ = 2.0f;
};
```

### 4.2 流水线并行

通过以下机制实现CPU和GPU之间的流水线并行：

1. **多流执行**：为CPU和GPU操作分配独立的流
2. **事件同步**：使用CUDA事件管理设备间依赖
3. **双缓冲**：为关键数据结构维护多个副本，实现生产者-消费者模式

```cpp
class PipelineExecutor {
public:
    // 初始化流水线
    void initialize(const ComputeGraph& graph);
    
    // 执行一次流水线迭代
    void step(const std::vector<Tensor<T>*>& inputs, std::vector<Tensor<T>*>& outputs);
    
    // 设置流水线深度
    void setPipelineDepth(int depth);
    
private:
    // 流水线阶段
    std::vector<PipelineStage> stages_;
    
    // 流和事件管理
    StreamManager stream_manager_;
    
    // 双缓冲管理
    BufferManager buffer_manager_;
};
```

## 5. 与现有算子库集成

### 5.1 适配现有算子

为了与现有的统一算子库无缝集成，我们将创建适配器节点：

```cpp
template <typename T>
class UnifiedOperatorNode : public OperatorNode {
public:
    UnifiedOperatorNode(OperatorType op_type, const std::vector<Tensor<T>*>& inputs)
        : op_type_(op_type), inputs_(inputs) {
        // 创建输出张量
        outputs_.resize(getOutputCount(op_type));
        for (int i = 0; i < outputs_.size(); ++i) {
            outputs_[i] = createOutputTensor(op_type, inputs, i);
        }
    }
    
    void execute(cudaStream_t stream = nullptr) override {
        // 获取适合当前设备的算子实现
        auto op = OperatorFactory<T>::getOperator(op_type_, getExecutionDevice() == Device::CUDA ? 
                                                 OperatorPlatform::CUDA : OperatorPlatform::CPU);
        
        // 准备参数
        prepareParameters();
        
        // 调用算子
        (*op)(param_ptrs_.data(), stream);
    }
    
private:
    OperatorType op_type_;
    std::vector<void*> param_ptrs_;  // 指向参数的指针数组
    
    // 准备算子参数
    void prepareParameters();
};
```

### 5.2 算子注册机制

扩展现有的算子注册机制，支持计算图节点：

```cpp
// 在算子工厂中添加创建节点的方法
template <typename T>
class OperatorFactory {
public:
    // 现有方法...
    
    // 创建算子节点
    static std::shared_ptr<OperatorNode> createNode(OperatorType type, const std::vector<Tensor<T>*>& inputs) {
        switch (type) {
            case OperatorType::MATMUL:
                return std::make_shared<MatMulNode<T>>(inputs);
            case OperatorType::ROPE:
                return std::make_shared<RopeNode<T>>(inputs);
            // 其他算子类型...
            default:
                return std::make_shared<UnifiedOperatorNode<T>>(type, inputs);
        }
    }
};
```

## 6. Qwen3 模型适配

### 6.1 模型结构表示

使用计算图表示 Qwen3 模型的前向计算过程：

```cpp
class Qwen3Model {
public:
    Qwen3Model(const ModelConfig& config);
    
    // 构建计算图
    void buildGraph();
    
    // 执行推理
    void forward(const Tensor<T>& input_ids, Tensor<T>* output, KVCache* kv_cache = nullptr);
    
private:
    // 模型参数
    std::unordered_map<std::string, Tensor<T>> params_;
    
    // 计算图
    ComputeGraph graph_;
    
    // 构建Transformer块
    NodeHandle buildTransformerBlock(NodeHandle input, int layer_idx);
    
    // 构建自注意力模块
    NodeHandle buildAttention(NodeHandle input, int layer_idx, KVCache* kv_cache);
    
    // 构建前馈网络
    NodeHandle buildMLP(NodeHandle input, int layer_idx);
};
```

### 6.2 KV缓存管理

针对 Qwen3 的 KV 缓存设计特殊的内存管理策略：

```cpp
class KVCacheManager {
public:
    // 分配KV缓存
    KVCache* allocate(int batch_size, int num_layers, int num_heads, int head_dim, int max_seq_len);
    
    // 扩展KV缓存
    void extend(KVCache* cache, int new_seq_len);
    
    // 释放KV缓存
    void free(KVCache* cache);
    
    // 设置卸载策略
    void setOffloadStrategy(KVCacheOffloadStrategy strategy);
    
private:
    // 缓存池
    std::vector<KVCache*> cache_pool_;
    
    // 卸载策略
    KVCacheOffloadStrategy offload_strategy_;
    
    // 内存管理器
    MemoryManager memory_manager_;
};
```

## 7. 性能优化

### 7.1 算子融合

识别并融合常见的算子组合，减少内存访问和内核启动开销：

1. **垂直融合**：融合连续执行的算子，如 Linear + GELU
2. **水平融合**：融合并行执行的相同算子，如多头注意力中的多个 MatMul

```cpp
class OperatorFuser {
public:
    // 分析计算图，识别可融合的模式
    void analyze(ComputeGraph& graph);
    
    // 执行融合
    void fuse(ComputeGraph& graph);
    
private:
    // 融合模式注册表
    std::vector<FusionPattern> fusion_patterns_;
    
    // 注册内置融合模式
    void registerBuiltinPatterns();
    
    // 检查节点是否匹配融合模式
    bool matchPattern(const std::vector<NodeHandle>& nodes, const FusionPattern& pattern);
};
```

### 7.2 内存优化

优化内存使用，减少设备间数据传输：

1. **原位操作**：尽可能使用原位算子，减少内存分配
2. **内存复用**：分析张量生命周期，复用不再需要的内存
3. **混合精度**：在合适的位置使用低精度计算

```cpp
class MemoryOptimizer {
public:
    // 分析内存使用模式
    void analyze(const ComputeGraph& graph);
    
    // 应用内存优化
    void optimize(ComputeGraph& graph);
    
    // 设置混合精度策略
    void setMixedPrecisionStrategy(MixedPrecisionStrategy strategy);
    
private:
    // 张量生命周期分析
    TensorLifetimeAnalyzer lifetime_analyzer_;
    
    // 内存分配计划
    MemoryAllocationPlan allocation_plan_;
    
    // 混合精度策略
    MixedPrecisionStrategy precision_strategy_;
};
```

## 8. 实现路线图

### 8.1 第一阶段：基础框架

1. 实现计算图数据结构和基本操作
2. 实现简单的设备选择策略
3. 适配现有算子库
4. 构建基本的 Qwen3 模型结构

### 8.2 第二阶段：内存管理与优化

1. 实现张量生命周期分析
2. 实现内存复用策略
3. 优化设备间数据传输
4. 实现 KV 缓存管理

### 8.3 第三阶段：高级特性

1. 实现动态卸载策略
2. 实现流水线并行
3. 实现算子融合
4. 添加混合精度支持

### 8.4 第四阶段：性能调优与测试

1. 针对不同硬件平台进行性能分析
2. 优化关键路径算子
3. 进行端到端性能测试
4. 与现有实现进行对比

## 9. 总结

本设计提出了一种基于计算图的 Qwen3 模型推理框架，支持异构计算和动态卸载。主要创新点包括：

1. **计算图抽象**：使用 DAG 表示模型计算，便于优化和调度
2. **异构计算**：支持算子在 CPU 和 GPU 之间灵活调度
3. **自动内存管理**：智能处理设备间数据传输和内存分配
4. **动态卸载**：根据运行时状态动态决定卸载策略
5. **流水线并行**：实现 CPU 和 GPU 之间的并行计算

通过这些技术，我们可以在保持高性能的同时，更灵活地适应不同的硬件环境和内存限制，为 Qwen3 模型提供更高效的推理支持。
