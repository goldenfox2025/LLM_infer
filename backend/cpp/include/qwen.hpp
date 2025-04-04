#pragma once
#include <cuda_bf16.h>  // For __nv_bfloat16 support
#include <unordered_map>
#include <stdexcept>
#include <string>


#include "base_model.hpp"
#include "cudaOP.cuh"
#include "inference.hpp"
#include "tensor.hpp"
#include "thread_pool.hpp"
constexpr int kNumStreams = 5;

template <typename T>
class QwenModel : public BaseModel {
 public:
  QwenModel(const std::unordered_map<std::string, Tensor<T>>& params,
            const std::unordered_map<std::string, int>& config);
  ~QwenModel() override;

  bool verify_params() const override;
  void print_model_info() const override;

  // Implementation of BaseModel interface:
  // 直接调用 CUDA 版本，并将 KVCacheBase* 动态转换为 KVCache<T>*
  uint32_t forward(const Tensor<uint32_t>* input, ThreadPool& thread_pool,
                   KVCacheBase* kv_cache, size_t top_k, float temperature,
                   float top_p, curandState* d_states = nullptr) override {
    KVCache<T>* typed_cache = dynamic_cast<KVCache<T>*>(kv_cache);

    return cuda_OP::sample(forward_cuda(input, typed_cache), temperature, top_p,
                           top_k, d_states);
  }
  uint32_t prefill(const Tensor<uint32_t>* input, ThreadPool& thread_pool,
                   KVCacheBase* kv_cache, size_t top_k, float temperature,
                   float top_p, curandState* d_states = nullptr) override {
    KVCache<T>* typed_cache = dynamic_cast<KVCache<T>*>(kv_cache);

    return cuda_OP::sample(prefill_cuda(input, typed_cache), temperature, top_p,
                           top_k, d_states);
  }

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

  // Additional getter methods for qwen_decode.cpp
  size_t get_n_heads() const { return n_heads_; }
  size_t get_hidden_size() const { return hidden_size_; }
  size_t get_intermediate_size() const { return intermediate_size_; }
  float get_rms_norm_eps() const { return rms_norm_eps_; }
  float get_rope_theta() const { return rope_theta_; }
  size_t get_vocab_size() const { return vocab_size_; }
  const std::unordered_map<std::string, Tensor<T>>& get_params() const {
    return params_;
  }

  // CUDA versions of forward and prefill.
  // Their implementations can be filled in later (currently as stubs mimicking
  // Llama).
  Tensor<T> forward_cuda(const Tensor<uint32_t>* input, KVCache<T>* kv_cache);
  Tensor<T> prefill_cuda(const Tensor<uint32_t>* input, KVCache<T>* kv_cache);

  // Device management
  QwenModel& cuda() override;
  QwenModel& cpu() override;
  Device device() const override { return device_; }

 private:
  cudaEvent_t eventQ, eventK, eventV;
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
  Device device_;

  std::array<cudaStream_t, kNumStreams> compute_streams_;
};

// 使用 extern template 声明已在别处定义的模板特化
// QwenModel<float> 特化
extern template class QwenModel<float>;
// QwenModel<__nv_bfloat16> 特化
extern template class QwenModel<__nv_bfloat16>;

// Helper function to convert weights from float to __nv_bfloat16
std::unordered_map<std::string, Tensor<__nv_bfloat16>> convert_weights_to_bf16(
    const std::unordered_map<std::string, Tensor<float>>& float_weights);
