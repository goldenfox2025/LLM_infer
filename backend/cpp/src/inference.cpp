#include "inference.hpp"

#include <cmath>
#include <functional>  // 添加以使用 std::function
#include <iostream>
#include <stdexcept>

#include "base_model.hpp"
#include "llama.hpp"
#include "operators.hpp"
#include "qwen.hpp"
// ------------------------
// KVCache 实现
// ------------------------
#include <iostream>
#include <stdexcept>

// 构造函数：分配连续内存并预计算各个位置的视图
#include <iostream>
#include <stdexcept>
template <typename T>
void debugPrintTensor(const Tensor<T>& tensor, const std::string& tensor_name,
                      size_t num_to_print = 10) {
  std::cout << "[Debug] " << tensor_name << ":\n";

  // 1) Print shape
  std::cout << "  shape: [";
  for (auto s : tensor.sizes()) {
    std::cout << s << " ";
  }
  std::cout << "]\n";

  // 2) Print strides
  std::cout << "  strides: [";
  for (auto st : tensor.strides()) {
    std::cout << st << " ";
  }
  std::cout << "]\n";

  // 3) Print device
  std::cout << "  device: ";
  if (tensor.device() == Device::CPU) {
    std::cout << "CPU";
  } else if (tensor.device() == Device::CUDA) {
    std::cout << "CUDA";
  } else {
    std::cout << "UNKNOWN";
  }
  std::cout << "\n";

  // 4) Print elements starting from offset 0
  size_t offset = 0;  // 从开始处打印
  size_t total_elements = tensor.numel();
  size_t n_print = std::min(num_to_print, total_elements - offset);

  std::cout << "  elements from offset " << offset << " (" << n_print
            << " element(s)): ";
  if (tensor.device() == Device::CPU) {
    const T* ptr = tensor.data_ptr();
    for (size_t i = 0; i < n_print; i++) {
      std::cout << ptr[offset + i] << " ";
    }
    std::cout << "\n";
  } else {
    // Copy from GPU to CPU, then print
    std::vector<T> host_buffer(n_print);
    cudaError_t err = cudaMemcpy(host_buffer.data(), tensor.data_ptr() + offset,
                                 n_print * sizeof(T), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      std::cout << "  [Error] cudaMemcpy failed\n";
      return;
    }
    for (size_t i = 0; i < n_print; i++) {
      std::cout << host_buffer[i] << " ";
    }
    std::cout << "\n";
  }
}
// 构造函数：分配连续内存并预先构造所有 slice（保存在一维 vector 中）
template <typename T>
KVCache<T>::KVCache(size_t n_layers, size_t max_seq_len, size_t head_dim,
                    Device device, size_t initial_size)
    : n_layers_(n_layers),
      max_seq_len_(max_seq_len),
      head_dim_(head_dim),
      current_len_(0),
      device_(device) {
  std::cout << "[KVCache::KVCache] 初始化 KVCache, n_layers=" << n_layers_
            << ", max_seq_len=" << max_seq_len_ << ", head_dim=" << head_dim_
            << ", device=" << (device_ == Device::CUDA ? "CUDA" : "CPU")
            << std::endl;

  // 分配连续内存，形状为 [n_layers, max_seq_len, head_dim]
  k_cache_contiguous_ =
      Tensor<T>({n_layers_, max_seq_len_, head_dim_}, device_);
  v_cache_contiguous_ =
      Tensor<T>({n_layers_, max_seq_len_, head_dim_}, device_);

  // （可选）初始化连续内存数据，比如 memset 为 0

  // 分配一维 vector 存储所有 slice
  k_cache_slices_.resize(n_layers_ * max_seq_len_);
  v_cache_slices_.resize(n_layers_ * max_seq_len_);
  for (size_t layer = 0; layer < n_layers_; layer++) {
    for (size_t pos = 0; pos < max_seq_len_; pos++) {
      size_t idx = layer * max_seq_len_ + pos;
      // 利用 slice 方法构造对应的 view，
      // 对于 k_cache_contiguous_，取范围 [layer, pos, 0] 到 [layer+1, pos+1,
      // head_dim_]
      k_cache_slices_[idx] = k_cache_contiguous_.slice(
          {layer, pos, 0}, {layer + 1, pos + 1, head_dim_});
      v_cache_slices_[idx] = v_cache_contiguous_.slice(
          {layer, pos, 0}, {layer + 1, pos + 1, head_dim_});
    }
  }

  // 设置初始有效 token 数（如果指定了初始大小）
  if (initial_size > 0) {
    if (initial_size > max_seq_len_) {
      throw std::runtime_error("Initial size cannot exceed max_seq_len");
    }
    current_len_ = initial_size;
  }
}

template <typename T>
void KVCache<T>::resize(size_t new_size) {
  if (new_size > max_seq_len_) {
    throw std::runtime_error("KVCache: Attempted to resize beyond max_seq_len");
  }
  if (new_size <= current_len_) {
    return;
  }
  current_len_ = new_size;
}

template <typename T>
void KVCache<T>::clear() {
  current_len_ = 0;
}
template <typename T>
Tensor<T>& KVCache<T>::k_cache(size_t layer, size_t pos) {
  if (layer >= n_layers_) {
    throw std::runtime_error("KVCache: Layer index out of range");
  }
  if (pos >= max_seq_len_) {
    throw std::runtime_error("KVCache: Position index out of range");
  }
  size_t idx = layer * max_seq_len_ + pos;
  return k_cache_slices_[idx];
}

template <typename T>
Tensor<T>& KVCache<T>::v_cache(size_t layer, size_t pos) {
  if (layer >= n_layers_) {
    throw std::runtime_error("KVCache: Layer index out of range");
  }
  if (pos >= max_seq_len_) {
    throw std::runtime_error("KVCache: Position index out of range");
  }
  size_t idx = layer * max_seq_len_ + pos;

  return v_cache_slices_[idx];
}

template <typename T>
KVCache<T>& KVCache<T>::cuda() {
  if (device_ == Device::CUDA) return *this;

  device_ = Device::CUDA;
  // 将连续内存移动到 CUDA 设备
  k_cache_contiguous_ = k_cache_contiguous_.cuda();
  v_cache_contiguous_ = v_cache_contiguous_.cuda();

  // 重新构造所有 slice（由于连续内存的 data pointer 更新）
  for (size_t layer = 0; layer < n_layers_; layer++) {
    for (size_t pos = 0; pos < max_seq_len_; pos++) {
      size_t idx = layer * max_seq_len_ + pos;
      k_cache_slices_[idx] = k_cache_contiguous_.slice(
          {layer, pos, 0}, {layer + 1, pos + 1, head_dim_});
      v_cache_slices_[idx] = v_cache_contiguous_.slice(
          {layer, pos, 0}, {layer + 1, pos + 1, head_dim_});
    }
  }
  return *this;
}

template <typename T>
KVCache<T>& KVCache<T>::cpu() {
  if (device_ == Device::CPU) return *this;

  device_ = Device::CPU;
  // 将连续内存移回 CPU
  k_cache_contiguous_ = k_cache_contiguous_.cpu();
  v_cache_contiguous_ = v_cache_contiguous_.cpu();

  // 重新构造所有 slice
  for (size_t layer = 0; layer < n_layers_; layer++) {
    for (size_t pos = 0; pos < max_seq_len_; pos++) {
      size_t idx = layer * max_seq_len_ + pos;
      k_cache_slices_[idx] = k_cache_contiguous_.slice(
          {layer, pos, 0}, {layer + 1, pos + 1, head_dim_});
      v_cache_slices_[idx] = v_cache_contiguous_.slice(
          {layer, pos, 0}, {layer + 1, pos + 1, head_dim_});
    }
  }
  return *this;
}

template <typename T>
std::pair<const Tensor<T>, const Tensor<T>> KVCache<T>::get_contiguous_tensor(
    size_t layer) const {
  Tensor<T> K = k_cache_contiguous_
                    .slice({layer, 0, 0}, {layer + 1, current_len_, head_dim_})
                    .squeeze(0);
  Tensor<T> V = v_cache_contiguous_
                    .slice({layer, 0, 0}, {layer + 1, current_len_, head_dim_})
                    .squeeze(0);

  return {K, V};
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
    std::cout
        << "[InferenceEngine::InferenceEngine] Moving InferenceEngine to CUDA"
        << std::endl;
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

  // debugPrintTensor(prefill_logits, "prefill_logits");
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

  // debugPrintTensor(last_logits, "last_logits");
  uint32_t next_token = OP::sample(&last_logits, temperature, top_p, top_k);

  // 通过回调函数将 token 传回给 Python
  if (next_token == model_->get_eos_token_id()) {
    return;
  }
  callback(next_token);

  std::vector<uint32_t> output = input_ids;
  output.push_back(next_token);

  // 继续生成直到达到 max_length 或遇到 eos
  while (output.size() < max_length) {
    // if (kv_cache_.size() >= kv_cache_.max_seq_len_) {
    //   callback(-1);  // 传递 -1 表示超出
    //   break;
    // }
    // auto start = std::chrono::high_resolution_clock::now();

    next_token =
        generate_next_token(thread_pool_, output, temperature, top_p, top_k);

    // 在生成 token 后记录结束时间
    // auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::milli> token_time = end - start;
    // std::cout << "生成 token 耗时: " << token_time.count() << " 毫秒"
    //           << std::endl;

    output.push_back(next_token);
    if (next_token == model_->get_eos_token_id()) {
      break;
    }
    callback(next_token);
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
  if (model_->device() == Device::CPU) {
    std::cout << "[InferenceEngine::cuda] Moving model to CUDA" << std::endl;
    model_->cuda();
  }

  std::cout << "[InferenceEngine::cuda] Moving KVCache to CUDA" << std::endl;
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
