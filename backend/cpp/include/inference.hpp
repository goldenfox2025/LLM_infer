#pragma once
#include <functional>
#include <memory>
#include <stdexcept>
#include <vector>

#include "tensor.hpp"
#include "thread_pool.hpp"
// 前向声明 BaseModel
class BaseModel;

// KVCache：用于存储每一层每个 token 的 Key 和 Value 张量
class KVCache {
 public:
  // n_layers：模型层数，max_seq_len：最大序列长度，
  // head_dim：每个缓存张量的元素数（通常为 n_kv_heads * dqkv）
  // initial_size：初始缓存 token 数（可选）
  // device: 指定 KVCache 所在的设备 (CPU or CUDA)
  KVCache(size_t n_layers, size_t max_seq_len, size_t head_dim,
          Device device = Device::CPU, size_t initial_size = 0);

  // 调整缓存长度，必须大于当前长度且不超过 max_seq_len
  void resize(size_t new_size);
  // 清空缓存（置当前长度为 0）
  void clear();
  // 当前缓存 token 数
  size_t size() const { return current_len_; }

  // 访问第 layer 层、位置 pos 的 K 缓存（返回引用）
  Tensor<float>& k_cache(size_t layer, size_t pos);
  // 访问第 layer 层、位置 pos 的 V 缓存（返回引用）
  Tensor<float>& v_cache(size_t layer, size_t pos);
  size_t max_seq_len_;

  // 移动 KVCache 到 CUDA 设备
  KVCache& cuda();
  // 移动 KVCache 到 CPU 设备 (可选，如果需要显式移回 CPU)
  KVCache& cpu();
  Device device() const { return device_; }

 private:
  std::vector<Tensor<float>> k_cache_;
  std::vector<Tensor<float>> v_cache_;
  size_t n_layers_;
  size_t head_dim_;
  size_t current_len_;
  Device device_;  // Track device for KVCache
};

class InferenceEngine {
 public:
  // 构造时传入共享的 BaseModel 实例
  // device: 指定 InferenceEngine 运行的设备 (CPU or CUDA)
  InferenceEngine(std::shared_ptr<BaseModel> model,
                  Device device = Device::CUDA);

  // 生成单个 token
  uint32_t generate_next_token(ThreadPool& thread_pool,
                               const std::vector<uint32_t>& input_ids,
                               float temperature = 1.0f, float top_p = 0.9f,
                               size_t top_k = 50);
  // 批量生成 token，直到达到 max_length 或遇到 eos
  void generate_with_callback(const std::vector<uint32_t>& input_ids,
                              size_t max_length, float temperature, float top_p,
                              size_t top_k,
                              std::function<void(uint32_t)> callback);
  // 重置推理状态（清空 KV 缓存）
  void reset();

  // 移动 InferenceEngine (及其模型和 KV Cache) 到 CUDA 设备
  InferenceEngine& cuda();
  // 移动 InferenceEngine (及其模型和 KV Cache) 到 CPU 设备 (可选)
  InferenceEngine& cpu();
  Device device() const { return device_; }

 private:
  ThreadPool thread_pool_;
  std::shared_ptr<BaseModel> model_;
  KVCache kv_cache_;
  Device device_;
};