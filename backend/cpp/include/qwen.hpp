#pragma once
#include <cuda_bf16.h>  // For __nv_bfloat16 support

#include <stdexcept>
#include <string>
#include <unordered_map>

#include "base_model.hpp"
#include "cudaOP.cuh"
#include "inference.hpp"
#include "operators/unified_operators.hpp"
#include "tensor.hpp"
#include "thread_pool.hpp"
#include "weight_tensor.hpp"

template <typename T>
class QwenModel : public BaseModel {
   public:
    QwenModel(const std::unordered_map<std::string, Tensor<T>>& params,
              const std::unordered_map<std::string, int>& config);

    // 带量化参数的构造函数
    QwenModel(const std::unordered_map<std::string, Tensor<T>>& params,
              const std::unordered_map<std::string, Tensor<int32_t>>& qweight_params,
              const std::unordered_map<std::string, Tensor<T>>& scales_params,
              const std::unordered_map<std::string, Tensor<int32_t>>& qzeros_params,
              const std::unordered_map<std::string, int>& config);
    ~QwenModel() override;

    bool verify_params() const override;
    void print_model_info() const override;
    void reset_async_preparation_state();  // 重置异步预准备状态（在新对话开始时调用）
    // Implementation of BaseModel interface:
    // 直接调用 CUDA 版本，并将 KVCacheBase* 动态转换为 KVCache<T>*
    uint32_t* forward(const Tensor<uint32_t>* input, ThreadPool& thread_pool, KVCacheBase* kv_cache, size_t top_k,
                      float temperature, float top_p, curandState* d_states = nullptr) override {
        KVCache<T>* typed_cache = dynamic_cast<KVCache<T>*>(kv_cache);

        if (use_cuda_graph_) {
            // 使用 CUDA 图执行
            // 延迟初始化CUDA图，使用真实的KV cache
            if (!graph_initialized_) {
                initialize_cuda_graph_with_kv_cache(typed_cache);
            }

            // 计算当前执行的参数
            size_t rope_offset = typed_cache->size() - 1;  // 当前token的位置索引
            size_t total_seq_len = typed_cache->size();    // flash attention应该看到包含当前token的完整数据

            if (last_kv_cache_size_ == 0 || (typed_cache->size() > last_kv_cache_size_ + 1)) {
                // 新轮对话开始，重置异步预准备状态
                reset_async_preparation_state();
                std::cout << "检测到新轮对话开始，KV cache从 " << last_kv_cache_size_ << " 跳跃到 "
                          << typed_cache->size() << std::endl;
            }

            // 将输入数据拷贝到图的固定输入张量
            cudaMemcpyAsync(graph_input_tensor_.data_ptr(), input->data_ptr(), input->numel() * sizeof(uint32_t),
                            cudaMemcpyDeviceToDevice, graph_stream_);
            // 更新记录的KV cache大小
            last_kv_cache_size_ = typed_cache->size();

            // 优化：检查是否已经预准备了当前执行
            if (!next_execution_prepared_) {
                // 如果没有预准备，则同步准备
                prepare_graph_execution(rope_offset, total_seq_len, 0, typed_cache);
            } else {
                // 如果已经预准备，等待预准备完成
                // 注意：KV地址已经在上次图执行后异步更新了，这里不需要再更新
                cudaEventSynchronize(prep_done_event_);
                next_execution_prepared_ = false;  // 重置标记
            }

            // 执行 CUDA 图
            cudaError_t result = cudaGraphLaunch(graph_exec_, graph_stream_);
            if (result != cudaSuccess) {
                throw std::runtime_error("Failed to launch CUDA graph: " + std::string(cudaGetErrorString(result)));
            }

            size_t next_rope_offset = typed_cache->size();  // 下一个token的位置
            update_graph_kv_addresses_async_for_next(typed_cache, next_rope_offset);
            // 优化：立即开始准备下一次执行的其他参数（异步）
            size_t next_total_seq_len = typed_cache->size() + 1;  // 下一次的序列长度
            prepare_next_graph_execution_async(next_rope_offset, next_total_seq_len, 0, typed_cache);
            //   cudaStreamSynchronize(graph_stream_);

            return cuda_OP::sample(std::move(graph_output_tensor_), temperature, top_p, top_k, d_states, graph_stream_);
        } else {
            // 使用常规CUDA推理
            return cuda_OP::sample(forward_cuda(input, typed_cache), temperature, top_p, top_k, d_states);
        }
    }
    uint32_t* prefill(const Tensor<uint32_t>* input, ThreadPool& thread_pool, KVCacheBase* kv_cache, size_t top_k,
                      float temperature, float top_p, curandState* d_states = nullptr) override {
        KVCache<T>* typed_cache = dynamic_cast<KVCache<T>*>(kv_cache);

        return cuda_OP::sample(prefill_cuda(input, typed_cache), temperature, top_p, top_k, d_states);
    }

    // Token generation
    std::vector<uint32_t> generate(const std::vector<uint32_t>& input_ids, size_t max_length, float temperature = 1.0f,
                                   float top_p = 0.9f, size_t top_k = 50);

    // Getter methods
    size_t get_n_layers() const override {
        return n_layers_;
    }
    size_t get_max_seq_len() const override {
        return max_position_embeddings_;
    }
    size_t get_head_dim() const override {
        return head_dim_;
    }
    size_t get_n_kv_heads() const override {
        return n_kv_heads_;
    }
    uint32_t get_eos_token_id() const override {
        return eos_token_id_;
    }
    size_t get_hidden_size() const override {
        return hidden_size_;
    }

    // Additional getter methods for qwen_decode.cpp
    size_t get_n_heads() const {
        return n_heads_;
    }
    size_t get_intermediate_size() const {
        return intermediate_size_;
    }
    float get_rms_norm_eps() const {
        return rms_norm_eps_;
    }
    float get_rope_theta() const {
        return rope_theta_;
    }
    size_t get_vocab_size() const {
        return vocab_size_;
    }
    int get_quant_type() const {
        return quant_type_;
    }
    const std::unordered_map<std::string, Tensor<T>>& get_params() const {
        return params_;
    }
    const std::unordered_map<std::string, Tensor<int32_t>>& get_qweight_params() const {
        return qweight_params_;
    }
    const std::unordered_map<std::string, Tensor<T>>& get_scales_params() const {
        return scales_params_;
    }
    const std::unordered_map<std::string, Tensor<int32_t>>& get_qzeros_params() const {
        return qzeros_params_;
    }

    // CUDA versions of forward and prefill.
    // Their implementations can be filled in later (currently as stubs mimicking
    // Llama).
    Tensor<T> forward_cuda(const Tensor<uint32_t>* input, KVCache<T>* kv_cache, const std::string& save_prefix = "");
    Tensor<T> prefill_cuda(const Tensor<uint32_t>* input, KVCache<T>* kv_cache);

    // 简化版本的前向传播，用于图优化开发
    // 特点：
    // 1. 只包含必要的算子，去除了复杂的优化逻辑
    // 2. 不使用KV缓存的复杂管理，简化注意力计算
    // 3. 统一使用operators_接口，便于后续图优化
    // 4. 移除了流并行和事件同步等优化
    // 5. 支持指定流执行，用于 CUDA 图捕获
    Tensor<T> forward_for_graph(const Tensor<uint32_t>* input, KVCache<T>* kv_cache, cudaStream_t stream = nullptr);

    // 获取权重（普通或量化）
    op::WeightTensor<T> get_weight(const std::string& key) {
        if (quant_type_ == 1) {
            // 尝试获取量化权重
            std::string qweight_key = key + ".qweight";
            std::string scales_key = key + ".scales";
            std::string qzeros_key = key + ".qzeros";

            auto qweight_it = qweight_params_.find(qweight_key);
            auto scales_it = scales_params_.find(scales_key);
            auto qzeros_it = qzeros_params_.find(qzeros_key);

            if (qweight_it != qweight_params_.end() && scales_it != scales_params_.end() &&
                qzeros_it != qzeros_params_.end()) {
                // 返回量化权重
                return op::WeightTensor<T>(&qweight_it->second, &scales_it->second, &qzeros_it->second, group_size_);
            }
        }

        auto weight_it = params_.find(key);
        if (weight_it != params_.end()) {
            return op::WeightTensor<T>(&weight_it->second);
        }
        // 尝试返回普通权重
        std::string weight_key = key + ".weight";
        weight_it = params_.find(weight_key);

        if (weight_it != params_.end()) {
            return op::WeightTensor<T>(&weight_it->second);
        }

        // 如果找不到权重，抛出更明确的错误
        throw std::runtime_error("Weight not found: " + key +
                                 (quant_type_ == 1 ? " (tried both quantized and regular)" : " (tried regular)"));
    }

    // Device management
    QwenModel& cuda() override;
    QwenModel& cpu() override;
    Device device() const override {
        return device_;
    }

    // CUDA图优化相关方法
    void initialize_graph_fixed_memory();    // 初始化图执行所需的固定内存
    void cleanup_graph_fixed_memory();       // 清理图执行的固定内存
    void update_rope_offset(size_t offset);  // 更新RoPE offset到固定内存
    void update_segment_info(size_t total_seq_len,
                             int layer_idx);  // 更新flash attention分段信息
    void extract_updateable_nodes();          // 从图中提取可更新的节点
    void update_graph_kv_addresses(KVCache<T>* kv_cache,
                                   size_t offset);  // 更新图中的KV复制目标地址
    void update_graph_kv_addresses_async_for_next(KVCache<T>* kv_cache,
                                                  size_t next_offset);  // 异步更新图节点参数为下一次执行准备
    void prepare_graph_execution(size_t rope_offset, size_t total_seq_len, int layer_idx,
                                 KVCache<T>* kv_cache);  // 在图执行前准备所有动态数据
    void prepare_next_graph_execution_async(size_t next_rope_offset, size_t next_total_seq_len, int layer_idx,
                                            KVCache<T>* kv_cache);  // 异步准备下一次图执行的动态数据

    void initialize_cuda_graph_with_kv_cache(KVCache<T>* kv_cache);  // 使用真实KV cache初始化CUDA图

    // 逐层输出保存功能
    void save_tensor_to_binary(const Tensor<T>& tensor,
                               const std::string& filename);  // 保存张量到二进制文件
    void save_uint32_tensor_to_binary(const Tensor<uint32_t>& tensor,
                                      const std::string& filename);           // 保存uint32张量到二进制文件
    void save_graph_tensors_after_execution(const std::string& save_prefix);  // 图执行后保存重要张量

   private:
    std::array<cudaEvent_t, 3> fa_done_events_;
    size_t vocab_size_;
    size_t n_layers_;
    size_t n_heads_;
    size_t n_kv_heads_;
    size_t hidden_size_;
    size_t head_dim_;
    size_t intermediate_size_;
    size_t max_position_embeddings_;
    uint32_t bos_token_id_;
    uint32_t eos_token_id_;
    float rms_norm_eps_;
    float rope_theta_;

    std::unordered_map<std::string, Tensor<T>> params_;
    std::unordered_map<std::string, Tensor<int32_t>> qweight_params_;  // 量化权重
    std::unordered_map<std::string, Tensor<T>> scales_params_;         // 缩放因子
    std::unordered_map<std::string, Tensor<int32_t>> qzeros_params_;   // 零点
    int quant_type_ = 0;                                               // 0: 非量化, 1: AWQ量化
    int group_size_ = 128;                                             // 量化分组大小
    Device device_;

    std::array<cudaStream_t, kNumStreams> compute_streams_;

    // 统一算子接口
    std::unique_ptr<op::UnifiedOperators<T>> operators_;

    // 推理模式控制 - 默认使用常规CUDA推理，手动修改此值来测试CUDA图
    bool use_cuda_graph_ = true;  // 改为true来启用CUDA图推理

    // CUDA 图执行相关成员
    cudaGraph_t cuda_graph_;
    cudaGraphExec_t graph_exec_;
    cudaStream_t graph_stream_;
    bool graph_initialized_;

    // 图执行所需的固定输入输出张量
    Tensor<uint32_t> graph_input_tensor_;
    Tensor<T> graph_output_tensor_;

    // 问题1解决方案：RoPE offset的固定内存
    size_t* d_rope_offset_;  // 设备端固定内存存储offset

    // 当前KV写入offset（用于图执行）
    size_t current_kv_offset_;

    // 问题2解决方案：matmul固定中间缓冲区
    std::vector<Tensor<T>> fixed_k_buffers_;  // 每层的K投影固定缓冲区
    std::vector<Tensor<T>> fixed_v_buffers_;  // 每层的V投影固定缓冲区

    // 图节点更新相关
    std::vector<cudaGraphNode_t> kv_copy_nodes_;  // KV复制节点列表，用于更新目标地址

    // KV地址更新优化：缓存节点参数
    std::vector<cudaMemcpy3DParms> cached_k_params_;  // 缓存的K节点参数
    std::vector<cudaMemcpy3DParms> cached_v_params_;  // 缓存的V节点参数
    bool kv_params_cached_;                           // 标记参数是否已缓存

    // 问题3解决方案：flash attention固定内存地址和分段信息
    int* d_segment_info_;                      // 设备端分段信息：[total_seq_len, branch_count,
                                               // branch_lengths...]
    T** d_output_ptrs_;                        // 设备端输出指针数组
    std::vector<Tensor<T>> fixed_fa_outputs_;  // 每层的flash attention输出固定内存

    // 异步预准备优化相关成员
    bool next_execution_prepared_;  // 标记下一次执行是否已经预准备完成
    cudaStream_t prep_stream_;      // 专用于预准备操作的CUDA流
    cudaEvent_t prep_done_event_;   // 预准备完成事件
    size_t last_kv_cache_size_;     // 记录上次执行时的KV cache大小，用于检测新轮对话
};

// 使用 extern template 声明已在别处定义的模板特化
// QwenModel<float> 特化
extern template class QwenModel<float>;
// QwenModel<__nv_bfloat16> 特化
extern template class QwenModel<__nv_bfloat16>;
