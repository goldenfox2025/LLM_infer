#pragma once
#include <array>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "base_model.hpp"
#include "common.hpp"
#include "cudaOP.cuh"
#include "inference.hpp"
#include "operators/unified_operators.hpp"
#include "tensor.hpp"
#include "thread_pool.hpp"

template <typename T>
class Qwen3Model : public BaseModel {
   public:
    Qwen3Model(const std::unordered_map<std::string, Tensor<T>>& params,
               const std::unordered_map<std::string, int>& config);

    // 带量化参数的构造函数
    Qwen3Model(const std::unordered_map<std::string, Tensor<T>>& params,
               const std::unordered_map<std::string, Tensor<int32_t>>& qweight_params,
               const std::unordered_map<std::string, Tensor<T>>& scales_params,
               const std::unordered_map<std::string, Tensor<int32_t>>& qzeros_params,
               const std::unordered_map<std::string, int>& config);
    ~Qwen3Model() override;

    bool verify_params() const override;
    void print_model_info() const override;

    // Implementation of BaseModel interface:
    uint32_t* forward(const Tensor<uint32_t>* input, ThreadPool& thread_pool, KVCacheBase* kv_cache, size_t top_k,
                      float temperature, float top_p, curandState* d_states = nullptr) override;
    uint32_t* prefill(const Tensor<uint32_t>* input, ThreadPool& thread_pool, KVCacheBase* kv_cache, size_t top_k,
                      float temperature, float top_p, curandState* d_states = nullptr) override;

    // Device management
    Qwen3Model& cuda() override;
    Qwen3Model& cpu() override;
    Device device() const override {
        return device_;
    }

    // Getters for model properties
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

    // Additional getter methods
    size_t get_n_heads() const {
        return n_heads_;
    }
    size_t get_hidden_size() const {
        return hidden_size_;
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

    // 获取权重（普通或量化）
    op::WeightTensor<T> get_weight(const std::string& key) {
        if (quant_type_ == 1) {
            // 尝试获取量化权重
            // 对于Qwen3模型，不需要添加.qweight后缀
            auto qweight_it = qweight_params_.find(key);
            auto scales_it = scales_params_.find(key);
            auto qzeros_it = qzeros_params_.find(key);

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

        // 如果找不到权重，抛出更明确的错误
        throw std::runtime_error("Weight not found: " + key +
                                 (quant_type_ == 1 ? " (tried both quantized and regular)" : " (tried regular)"));
    }

    // CUDA versions of forward and prefill
    Tensor<T> forward_cuda(const Tensor<uint32_t>* input, KVCache<T>* kv_cache);
    Tensor<T> prefill_cuda(const Tensor<uint32_t>* input, KVCache<T>* kv_cache);

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
};

// 使用 extern template 声明已在别处定义的模板特化
// Qwen3Model<__nv_bfloat16> 特化
extern template class Qwen3Model<__nv_bfloat16>;
