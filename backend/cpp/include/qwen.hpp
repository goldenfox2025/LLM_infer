#pragma once
#include <cuda_bf16.h>  // For __nv_bfloat16 support
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "base_model.hpp"
#include "inference.hpp"
#include "tensor.hpp"
#include "thread_pool.hpp"

// Base QwenModel implementation with templated precision
template <typename T>
class QwenModel : public BaseModel {
 public:
  QwenModel(const std::unordered_map<std::string, Tensor<T>>& params,
            const std::unordered_map<std::string, int>& config);
  
  bool verify_params() const override;
  void print_model_info() const override;

  // Implementation of BaseModel interface
  Tensor<float> forward(const Tensor<uint32_t>* input, ThreadPool& thread_pool,
                      KVCache* kv_cache) override;
  Tensor<float> prefill(const Tensor<uint32_t>* input, ThreadPool& thread_pool,
                      KVCache* kv_cache) override;
  
  // Qwen-specific implementations
  Tensor<T> prefill_internal(const Tensor<uint32_t>* input, ThreadPool& thread_pool,
                           KVCache* kv_cache);
  Tensor<T> prefill_cpu(const Tensor<uint32_t>* input, KVCache* kv_cache,
                       ThreadPool& thread_pool);
  Tensor<T> prefill_cuda(const Tensor<uint32_t>* input, KVCache* kv_cache);
  Tensor<T> forward_internal(const Tensor<uint32_t>* input, ThreadPool& thread_pool,
                           KVCache* kv_cache);
  Tensor<T> forward_cpu(const Tensor<uint32_t>* input, ThreadPool& thread_pool,
                       KVCache* kv_cache);
  Tensor<T> forward_cuda(const Tensor<uint32_t>* input, KVCache* kv_cache);

  // Token generation
  std::vector<uint32_t> generate(const std::vector<uint32_t>& input_ids,
                                 size_t max_length, float temperature = 1.0f,
                                 float top_p = 0.9f, size_t top_k = 50);

  // Getter methods
  size_t get_n_layers() const override { return n_layers_; }
  size_t get_max_seq_len() const override { return max_position_embeddings_; }
  size_t get_head_dim() const override { return head_dim_; }
  size_t get_n_kv_heads() const override { return n_kv_heads_; }
  uint32_t get_eos_token_id() const override { return eos_token_id_; }

  // Device management
  QwenModel& cuda() override;
  QwenModel& cpu() override;
  Device device() const override { return device_; }

 private:
  // Basic parameters
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

  // Model parameters
  std::unordered_map<std::string, Tensor<T>> params_;
  Device device_;
};

// 使用extern template声明已在别处定义的模板特化
// QwenModel<float>特化
extern template class QwenModel<float>;

// QwenModel<__nv_bfloat16>特化
extern template class QwenModel<__nv_bfloat16>;

// Helper function to convert weights from float to __nv_bfloat16
std::unordered_map<std::string, Tensor<__nv_bfloat16>> convert_weights_to_bf16(
    const std::unordered_map<std::string, Tensor<float>>& float_weights);

// 添加extern template声明，表示这些实例化将在别处定义
// 这可以防止包含此头文件的其他代码隐式实例化这些模板
extern template class QwenModel<float>;
extern template class QwenModel<__nv_bfloat16>;
