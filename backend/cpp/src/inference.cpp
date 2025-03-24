#include "inference.hpp"

#include <cmath>
#include <functional>  // 添加以使用 std::function
#include <iostream>
#include <stdexcept>

#include "base_model.hpp"
#include "llama.hpp"
#include "operators.hpp"

// ------------------------
// KVCache 实现
// ------------------------
template <typename T>
KVCache<T>::KVCache(size_t n_layers, size_t max_seq_len, size_t head_dim,
                    Device device, size_t initial_size)
    : n_layers_(n_layers),
      max_seq_len_(max_seq_len),
      head_dim_(head_dim),
      current_len_(0),
      device_(device) {  // 初始化 device_
  std::cout << "[KVCache::KVCache] 初始化 KVCache, n_layers=" << n_layers_
            << ", max_seq_len=" << max_seq_len_ << ", head_dim=" << head_dim_
            << ", device=" << (device_ == Device::CUDA ? "CUDA" : "CPU")
            << std::endl;

  // 分配每一层每个位置的缓存（总共 n_layers * max_seq_len 个 Tensor）
  k_cache_.resize(n_layers_ * max_seq_len_);
  v_cache_.resize(n_layers_ * max_seq_len_);
  for (size_t i = 0; i < n_layers_ * max_seq_len_; i++) {
    // 初始化一个大小为 head_dim 的数据向量，全部置 0
    std::vector<T> k_data(head_dim_, static_cast<T>(0.0f));
    std::vector<T> v_data(head_dim_, static_cast<T>(0.0f));
    // 使用数据向量构造 Tensor，形状为 [1, head_dim]，并指定设备
    k_cache_[i] = Tensor<T>(std::move(k_data), {1, head_dim_}, device_);
    v_cache_[i] = Tensor<T>(std::move(v_data), {1, head_dim_}, device_);
  }
  // 如果指定了初始大小，则设置当前长度
  if (initial_size > 0) {
    if (initial_size > max_seq_len_) {
      throw std::runtime_error("Initial size cannot exceed max_seq_len");
    }
    current_len_ = initial_size;
    // std::cout << "[KVCache::KVCache] 设置初始 current_len: " << current_len_
    //           << std::endl;
  }
}

template <typename T>
void KVCache<T>::resize(size_t new_size) {
  // std::cout << "[KVCache::resize] 请求 resize from " << current_len_ << " to
  // "
  //           << new_size << std::endl;
  if (new_size > max_seq_len_) {
    throw std::runtime_error("KVCache: Attempted to resize beyond max_seq_len");
  }
  if (new_size <= current_len_) {
    // std::cout << "[KVCache::resize] new_size <= current_len, 无需调整"
    //           << std::endl;
    return;
    // throw std::runtime_error(
    //     "KVCache: New size must be larger than current size");
  }
  current_len_ = new_size;
  // std::cout << "[KVCache::resize] 成功调整 current_len 为 " << current_len_
  //           << std::endl;
}

template <typename T>
void KVCache<T>::clear() {
  // std::cout << "[KVCache::clear] 清空 KVCache" << std::endl;
  current_len_ = 0;
}

template <typename T>
Tensor<T>& KVCache<T>::k_cache(size_t layer, size_t pos) {
  // std::cout << "[KVCache::k_cache] 访问 layer=" << layer << ", pos=" << pos
  //           << std::endl;
  if (layer >= n_layers_) {
    throw std::runtime_error("KVCache: Layer index out of range");
  }
  if (pos >= max_seq_len_) {
    throw std::runtime_error("KVCache: Position index out of range");
  }
  return k_cache_[layer * max_seq_len_ + pos];
}

template <typename T>
Tensor<T>& KVCache<T>::v_cache(size_t layer, size_t pos) {
  // std::cout << "[KVCache::v_cache] 访问 layer=" << layer << ", pos=" << pos
  //           << std::endl;
  if (layer >= n_layers_) {
    throw std::runtime_error("KVCache: Layer index out of range");
  }
  if (pos >= max_seq_len_) {
    throw std::runtime_error("KVCache: Position index out of range");
  }
  return v_cache_[layer * max_seq_len_ + pos];
}

template <typename T>
KVCache<T>& KVCache<T>::cuda() {
  if (device_ == Device::CUDA) return *this;

  device_ = Device::CUDA;
  for (auto& t : k_cache_) {
    t = t.cuda();
  }
  for (auto& t : v_cache_) {
    t = t.cuda();
  }
  return *this;
}

template <typename T>
KVCache<T>& KVCache<T>::cpu() {
  if (device_ == Device::CPU) return *this;

  device_ = Device::CPU;
  for (auto& t : k_cache_) {
    t = t.cpu();
  }
  for (auto& t : v_cache_) {
    t = t.cpu();
  }
  return *this;
}

// 显式实例化模板类
template class KVCache<float>;
template class KVCache<__nv_bfloat16>;

// ------------------------
// InferenceEngine 实现
// ------------------------
template <typename T>
InferenceEngine<T>::InferenceEngine(std::shared_ptr<BaseModel> model,
                                    Device device)
    : model_(model),
      // 使用模型参数初始化 KVCache，并指定设备
      kv_cache_(model_->get_n_layers(), model_->get_max_seq_len(),
                model_->get_head_dim() * model_->get_n_kv_heads(), device),
      thread_pool_(4),
      device_(device) {  // 初始化 device_
  // std::cout
  //     << "[InferenceEngine::InferenceEngine] 初始化 InferenceEngine, device="
  //     << (device_ == Device::CUDA ? "CUDA" : "CPU") << std::endl;
  if (device_ == Device::CUDA) {
    this->cuda();
  }
}

template <typename T>
uint32_t InferenceEngine<T>::generate_next_token(
    ThreadPool& thread_pool, const std::vector<uint32_t>& input_ids,
    float temperature, float top_p, size_t top_k) {
  // std::cout << "[InferenceEngine::generate_next_token] 开始生成下一个 token"
  //           << std::endl;
  // 构造输入张量，取 input_ids 中最后一个 token, 放置在正确的设备上
  Tensor<uint32_t> input({input_ids.back()}, {1}, device_);
  // std::cout << "[InferenceEngine::generate_next_token] 输入 token: "
  //           << input_ids.back() << std::endl;

  // 更新 KV 缓存长度（为新 token 分配缓存空间）
  try {
    kv_cache_.resize(kv_cache_.size() + 1);
    // std::cout << "[InferenceEngine::generate_next_token] KVCache 成功扩容"
    //           << std::endl;
  } catch (const std::runtime_error& e) {
    std::cerr << "Error resizing KV cache: " << e.what() << std::endl;
    throw;
  }

  // 前向计算，传入 KVCache 的地址
  Tensor<float> logits = model_->forward(&input, thread_pool, &kv_cache_);
  // std::cout << "[InferenceEngine::generate_next_token] 前向计算完成"
  //           << std::endl;

  // if (logits.device() != device_) {
  //   if (device_ == Device::CUDA) {
  //     logits.cuda();  // 确保 logits 在正确的设备上
  //   } else {
  //     logits.cpu();
  //   }
  //   // std::cout
  //   // << "[InferenceEngine::generate_next_token] 将 logits 移动到正确的设备"
  //   // << std::endl;
  // }

  // 根据 logits 采样下一个 token
  uint32_t next_token = OP::sample(&logits, temperature, top_p, top_k);
  // std::cout << "[InferenceEngine::generate_next_token] 采样得到 token: "
  //           << next_token << std::endl;

  return next_token;
}

// 在 InferenceEngine 类的实现文件（例如 inference.cpp）中增加：
template <typename T>
void InferenceEngine<T>::generate_with_callback(
    const std::vector<uint32_t>& input_ids, size_t max_length,
    float temperature, float top_p, size_t top_k,
    std::function<void(uint32_t)> callback) {
  // 如果需要从头开始，可清空缓存
  // kv_cache_.clear();
  // 让 KVCache 扩容到容纳 input_ids.size() 个位置
  // std::cout << "[InferenceEngine::generate_with_callback] 扩容 KVCache 到 "
  // << kv_cache_.size() + input_ids.size() << " 个位置" << std::endl;
  try {
    kv_cache_.resize(kv_cache_.size() + input_ids.size());
  } catch (const std::runtime_error& e) {
    std::cerr << "Error resizing KV cache (prefill): " << e.what() << std::endl;
    throw;
  }
  // std::cout << "[InferenceEngine::generate_with_callback] KVCache 扩容完成"
  // << std::endl;
  // 输入 tensor 也放在正确的设备上
  Tensor<uint32_t> input_tensor(std::vector<uint32_t>(input_ids),
                                {input_ids.size()}, device_);
  // std::cout << "[InferenceEngine::generate_with_callback] 输入 tensor "
  //              "放置在正确设备上 "
  //           << std::endl;
  // 调用 prefill，一次性处理全部 input_ids
  Tensor<float> prefill_logits =
      model_->prefill(&input_tensor, thread_pool_, &kv_cache_);

  // std::cout << "[InferenceEngine::prefill] " << std::endl;

  // prefill_logits.shape = [seq_len, vocab_size]
  size_t seq_len = input_ids.size();
  if (seq_len == 0) {
    return;
  }

  // 从 prefill 的最后一行 logits 中采样第一个生成 token
  const size_t vocab_size = prefill_logits.sizes()[1];
  std::vector<float> last_row_data(vocab_size, 0.f);
  const float* prefill_ptr =
      prefill_logits.data_ptr() + (seq_len - 1) * vocab_size;
  std::copy(prefill_ptr, prefill_ptr + vocab_size, last_row_data.begin());
  Tensor<float> last_logits(std::move(last_row_data), {1, vocab_size},
                            Device::CPU);  // 强制使用CPU设备

  uint32_t next_token = OP::sample(&last_logits, temperature, top_p, top_k);

  // 通过回调函数将 token 传回给 Python
  callback(next_token);
  if (next_token == model_->get_eos_token_id()) {
    return;
  }

  std::vector<uint32_t> output = input_ids;
  output.push_back(next_token);

  // 继续生成直到达到 max_length 或遇到 eos
  while (output.size() < max_length) {
    // if (kv_cache_.size() >= kv_cache_.max_seq_len_) {
    //   callback(-1);  // 传递 -1 表示超出
    //   break;
    // }
    next_token =
        generate_next_token(thread_pool_, output, temperature, top_p, top_k);
    output.push_back(next_token);
    callback(next_token);
    if (next_token == model_->get_eos_token_id()) {
      break;
    }
  }
}
template <typename T>
void InferenceEngine<T>::reset() {
  kv_cache_.clear();
}
template <typename T>
InferenceEngine<T>& InferenceEngine<T>::cuda() {
  if (device_ == Device::CUDA) {
    return *this;
  }
  model_->cuda();
  kv_cache_.cuda();
  device_ = Device::CUDA;
  return *this;
}
template <typename T>
InferenceEngine<T>& InferenceEngine<T>::cpu() {
  if (device_ == Device::CPU) {
    return *this;
  }
  model_->cpu();
  kv_cache_.cpu();
  device_ = Device::CPU;
  return *this;
}

// 显式实例化模板类 InferenceEngine
template class InferenceEngine<float>;
template class InferenceEngine<__nv_bfloat16>;
