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

// Sample模式枚举
enum class SampleMode {
    GPU,                    // 纯GPU sample（默认，性能最佳）
    CPU,                    // 纯CPU sample（保留用于测试）
    GPU_WITH_ASYNC_PREPARE  // GPU sample + 异步prepare_next（推荐用于EOS重叠）
};

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
    uint32_t* forward(const Tensor<uint32_t>* input, ThreadPool& thread_pool, KVCacheBase* kv_cache, size_t top_k,
                      float temperature, float top_p, curandState* d_states = nullptr) override {
        KVCache<T>* typed_cache = dynamic_cast<KVCache<T>*>(kv_cache);

        // 1. 执行GPU forward，获得logits
        Tensor<T> logits;
        if (use_cuda_graph_) {
            // CUDA图路径的logits获取
            logits = forward_for_graph_logits_only(input, typed_cache);

            // 图推理结束后批量应用预计算的offset
            apply_prepared_offsets();
        } else {
            // 常规CUDA路径的logits获取
            logits = forward_logits_only(input, typed_cache);
        }

        // 2. 使用新的统一sample接口
        if (use_cuda_graph_) {
            return sample_unified(logits, temperature, top_p, top_k, typed_cache, d_states, graph_stream_);
        } else {
            return sample_unified(logits, temperature, top_p, top_k, typed_cache, d_states);
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

    Tensor<T> forward_for_graph(const Tensor<uint32_t>* input, KVCache<T>* kv_cache, cudaStream_t stream = nullptr);

    // CPU sample相关方法
    Tensor<T> forward_logits_only(const Tensor<uint32_t>* input, KVCache<T>* kv_cache);
    Tensor<T> forward_for_graph_logits_only(const Tensor<uint32_t>* input, KVCache<T>* kv_cache);
    uint32_t sample_cpu(const Tensor<T>& gpu_logits, float temperature, float top_p, size_t top_k);
    uint32_t* sample_with_metadata_update(const Tensor<T>& logits, float temperature, float top_p, size_t top_k,
                                          KVCache<T>* kv_cache);
    uint32_t* allocate_gpu_result(uint32_t result);

    // 新的sample架构方法
    void set_sample_mode(SampleMode mode);
    uint32_t* sample_unified(const Tensor<T>& logits, float temperature, float top_p, size_t top_k,
                             KVCache<T>* kv_cache, curandState* d_states = nullptr, cudaStream_t stream = nullptr);
    uint32_t* sample_with_cpu_only(const Tensor<T>& logits, float temperature, float top_p, size_t top_k);

    // 新的offset优化方法
    void compute_next_offsets_async(int offset);
    void apply_prepared_offsets();
    void initialize_offset_cache();

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
    void prepare_graph_execution(size_t rope_offset, size_t total_seq_len, int layer_idx, KVCache<T>* kv_cache,
                                 cudaStream_t stream, int pingpong_index = 0);  // 在图执行前准备所有动态数据

    void initialize_cuda_graph_with_kv_cache(KVCache<T>* kv_cache);  // 使用真实KV cache初始化CUDA图

    // 逐层输出保存功能
    void save_tensor_to_binary(const Tensor<T>& tensor,
                               const std::string& filename);  // 保存张量到二进制文件
    void save_uint32_tensor_to_binary(const Tensor<uint32_t>& tensor,
                                      const std::string& filename);           // 保存uint32张量到二进制文件
    void save_graph_tensors_after_execution(const std::string& save_prefix);  // 图执行后保存重要张量

    // RoPE相关方法
    const Tensor<float>& get_rope_sin_cos_cache() const {
        return rope_sin_cos_cache_;
    }  // 获取RoPE sin/cos缓存
    bool has_rope_cache() const {
        return rope_sin_cos_cache_.numel() > 0;
    }  // 检查是否已初始化RoPE缓存

   private:
    int pingpong_index_ = 0;
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

    // CPU算子接口（用于sample操作）
    std::unique_ptr<op::UnifiedOperators<T>> cpu_operators_;

    // Sample模式选择
    SampleMode sample_mode_ = SampleMode::GPU;  // 默认使用GPU sample，性能最佳

    // 批量offset优化：预计算并缓存下一轮的offset值
    std::vector<int> next_batch_offsets_;
    bool offsets_prepared_ = false;

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

    // RoPE预计算的sin/cos缓存
    Tensor<float> rope_sin_cos_cache_;  // 存储预计算的sin/cos值，形状为[max_seq_len, head_dim]

    size_t* d_rope_offset_;  // 设备端固定内存存储offset
    int* d_offset_array_;    // 设备端连续offset数组，所有层共享，通过索引访问

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
