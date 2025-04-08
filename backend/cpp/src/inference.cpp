#include "inference.hpp"

#include <chrono>  // 用于计时
#include <cmath>
#include <condition_variable>
#include <exception>
#include <functional>  // 添加以使用 std::function
#include <iostream>
#include <mutex>
#include <optional>
#include <queue>
#include <stdexcept>
#include <variant>

#include "base_model.hpp"
#include "cudaOP.cuh"
#include "llama.hpp"
#include "operators.hpp"
#include "qwen.hpp"
// ------------------------
// KVCache 实现
// ------------------------
#include <iostream>
#include <stdexcept>

// 构造函数：分配连续内存并预计算各个位置的视图
#include <pybind11/stl.h>

#include <iostream>
#include <stdexcept>
template <typename T>
class ThreadSafeQueue {
 public:
  void push(T value) {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push(std::move(value));
    cv_.notify_one();
  }

  T pop() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return !queue_.empty(); });
    T value = std::move(queue_.front());
    queue_.pop();
    return value;
  }
  // ... (其他成员函数，如 empty, clear, try_pop 可选) ...

 private:
  mutable std::mutex mutex_;
  std::queue<T> queue_;
  std::condition_variable cv_;
};

// 定义用于结果队列的类型
enum class Signal { EndOfStream };  // 定义 Signal 枚举
using GenerationResult =
    std::variant<uint32_t, Signal, std::exception_ptr>;  // 定义类型别名
namespace py = pybind11;
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
    cudaMalloc(&d_states, sizeof(curandState));
    cuda_OP::init_curand(d_states, 42, 0);
    std::cout
        << "[InferenceEngine::InferenceEngine] Moving InferenceEngine to CUDA"
        << std::endl;
    this->cuda();
  }
}

template <typename T>
uint32_t InferenceEngine<T>::generate_next_token(ThreadPool& thread_pool,
                                                 uint32_t input_ids,
                                                 float temperature, float top_p,
                                                 size_t top_k) {
  // std::cout << "[InferenceEngine::generate_next_token] 开始生成下一个 token"
  //           << std::endl;
  // 构造输入张量，取 input_ids 中最后一个 token, 放置在正确的设备上

  Tensor<uint32_t> input({input_ids}, {1}, device_);
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
  uint32_t next_token;
  // 前向计算，传入 KVCache 的地址
  if (device_ == Device::CUDA) {
    next_token = model_->forward(&input, thread_pool, &kv_cache_, top_k,
                                 temperature, top_p, d_states);
  } else {
    next_token = model_->forward(&input, thread_pool, &kv_cache_, top_k,
                                 temperature, top_p);
  }

  return next_token;
}

// template <typename T>
// void InferenceEngine<T>::generate_with_callback(
//     const std::vector<uint32_t>& input_ids, size_t max_length,
//     float temperature, float top_p, size_t top_k,
//     std::function<void(uint32_t)> callback) {
//   // 如果需要从头开始，可清空缓存
//   // kv_cache_.clear();
//   // 让 KVCache 扩容到容纳 input_ids.size() 个位置
//   // std::cout << "[InferenceEngine::generate_with_callback] 扩容 KVCache 到
//   // "
//   // << kv_cache_.size() + input_ids.size() << " 个位置" << std::endl;
//   try {
//     kv_cache_.resize(kv_cache_.size() + input_ids.size());
//   } catch (const std::runtime_error& e) {
//     std::cerr << "Error resizing KV cache (prefill): " << e.what() <<
//     std::endl; throw;
//   }

//   Tensor<uint32_t> input_tensor(std::vector<uint32_t>(input_ids),
//                                 {input_ids.size()}, device_);

//   uint32_t next_token;
//   if (device_ == Device::CPU)
//     next_token = model_->prefill(&input_tensor, thread_pool_, &kv_cache_,
//     top_k,
//                                  temperature, top_p);
//   else {
//     next_token = model_->prefill(&input_tensor, thread_pool_, &kv_cache_,
//     top_k,
//                                  temperature, top_p, d_states);
//   }

//   if (next_token == model_->get_eos_token_id()) {
//     return;
//   }
//   {
//     py::gil_scoped_release release;  // 释放 GIL
//     callback(next_token);
//   }
//   std::vector<uint32_t> output = input_ids;
//   output.push_back(next_token);

//   // 继续生成直到达到 max_length 或遇到 eos
//   while (output.size() < max_length) {
//     next_token =
//         generate_next_token(thread_pool_, output, temperature, top_p, top_k);

//     output.push_back(next_token);
//     if (next_token == model_->get_eos_token_id()) {
//       break;
//     }
//     {
//       py::gil_scoped_release release;  // 释放 GIL
//       callback(next_token);
//     }
//   }
// }
namespace py = pybind11;

template <typename T>
void InferenceEngine<T>::generate_with_callback(
    const std::vector<uint32_t>& input_ids, size_t max_length,
    float temperature, float top_p, size_t top_k,
    std::function<void(uint32_t)> callback) {
  ThreadSafeQueue<GenerationResult> result_queue;

  std::thread generation_thread([&, this, input_ids_copy = input_ids]() {
    try {
      // --- 在 try 块开始处声明 next_token ---
      uint32_t next_token;

      // --- 在移动 input_ids_copy 之前获取其大小 ---
      size_t input_size = input_ids_copy.size();

      // 统计 prefill 开始时间
      auto prefill_start = std::chrono::high_resolution_clock::now();

      // --- KV Cache Resize ---
      try {
        kv_cache_.resize(kv_cache_.size() + input_size);
      } catch (const std::runtime_error& e) {
        std::cerr << "Error resizing KV cache (prefill): " << e.what()
                  << std::endl;
        throw;
      }

      {
        std::vector<uint32_t> prefill_input =
            input_ids_copy;  // 创建真正的拷贝给 Tensor
        Tensor<uint32_t> input_tensor(std::move(prefill_input), {input_size},
                                      this->device_);

        if (this->device_ == Device::CPU) {
          next_token = this->model_->prefill(&input_tensor, this->thread_pool_,
                                             &this->kv_cache_, top_k,
                                             temperature, top_p);
        } else {
          next_token = this->model_->prefill(
              &input_tensor, this->thread_pool_, &this->kv_cache_, top_k,
              temperature, top_p, this->d_states);
        }
      }

      // 统计 prefill 结束时间并计算耗时
      auto prefill_end = std::chrono::high_resolution_clock::now();
      auto prefill_elapsed_ms =
          std::chrono::duration_cast<std::chrono::milliseconds>(prefill_end -
                                                                prefill_start)
              .count();

      // 如果是 EOS，就直接返回
      if (next_token == this->model_->get_eos_token_id()) {
        result_queue.push(Signal::EndOfStream);
        // 打印 prefill 耗时
        std::cout << std::endl;
        std::cout << "[prefill 耗时] " << prefill_elapsed_ms << " ms"
                  << std::endl;
        std::cout << "[decode 耗时] 0 ms" << std::endl;
        std::cout << std::endl;
        return;
      }

      // 首先将 prefill 得到的第一个 token 推入队列
      result_queue.push(next_token);

      size_t current_total_length = input_size + 1;
      uint32_t last_token = next_token;

      // 统计 decode（也就是正式推理循环）的开始时间
      // auto decode_start = std::chrono::high_resolution_clock::now();

      // 我们可以再开一个累加器，用于累积每次 step 的耗时
      // long long total_decode_elapsed_ms = 0;

      while (current_total_length < max_length) {
        // 每次 decode 一个 token，可以统计单次 forward 的耗时
        // auto step_start = std::chrono::high_resolution_clock::now();

        next_token = this->generate_next_token(this->thread_pool_, last_token,
                                               temperature, top_p, top_k);

        // auto step_end = std::chrono::high_resolution_clock::now();
        // auto step_elapsed_ms =
        //     std::chrono::duration_cast<std::chrono::milliseconds>(step_end -
        //                                                           step_start)
        //         .count();
        // total_decode_elapsed_ms += step_elapsed_ms;

        last_token = next_token;
        current_total_length++;

        bool is_eos = (next_token == this->model_->get_eos_token_id());

        if (is_eos) {
          result_queue.push(Signal::EndOfStream);

          // decode 结束时间
          // auto decode_end = std::chrono::high_resolution_clock::now();
          // auto decode_elapsed_ms =
          //     std::chrono::duration_cast<std::chrono::milliseconds>(
          //         decode_end - decode_start)
          //         .count();

          // std::cout << std::endl;
          // std::cout << "[prefill 耗时] " << prefill_elapsed_ms << " ms"
          //           << std::endl;
          // std::cout << "[decode 循环总耗时(循环外计算)] " <<
          // decode_elapsed_ms
          //           << " ms" << std::endl;
          // std::cout << "[decode 每 step 累计耗时(循环内累加)] "
          //           << total_decode_elapsed_ms << " ms" << std::endl;
          // std::cout << std::endl;
          return;
        }
        result_queue.push(next_token);
      }

      // 如果正常跑完 max_length
      result_queue.push(Signal::EndOfStream);

      // 统计 decode 完成时间
      auto decode_end = std::chrono::high_resolution_clock::now();
      auto decode_elapsed_ms =
          std::chrono::duration_cast<std::chrono::milliseconds>(decode_end -
                                                                decode_start)
              .count();

      // 打印时间信息
      std::cout << "[prefill 耗时] " << prefill_elapsed_ms << " ms"
                << std::endl;
      std::cout << "[decode 循环总耗时(循环外计算)] " << decode_elapsed_ms
                << " ms" << std::endl;
      std::cout << "[decode 每 step 累计耗时(循环内累加)] "
                << total_decode_elapsed_ms << " ms" << std::endl;

    } catch (...) {
      // 如果发生异常，把异常推入队列
      result_queue.push(std::current_exception());
    }
  });
  try {
    while (true) {
      GenerationResult result = result_queue.pop();
      bool should_break = false;
      std::visit(
          [&](auto&& arg) {
            using Type = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<Type, uint32_t>) {
              uint32_t token;

              token = arg;

              try {
                py::gil_scoped_release release;
                callback(token);
              } catch (const py::error_already_set& e) {
                std::cerr << "Python error in callback: " << e.what()
                          << std::endl;
                // 将 Python 异常包装成 C++ 异常并抛出
                throw std::runtime_error("Python callback failed: " +
                                         std::string(e.what()));
              } catch (const std::exception& e) {  // 捕获标准 C++ 异常
                std::cerr << "C++ error in callback: " << e.what() << std::endl;
                throw;  // 重新抛出
              } catch (...) {
                std::cerr << "Unknown error during callback execution."
                          << std::endl;
                throw;  // 重新抛出未知异常
              }
            } else if constexpr (std::is_same_v<Type, Signal>) {
              if (arg == Signal::EndOfStream) {
                should_break = true;  // 收到结束信号，准备退出循环
              }
            } else if constexpr (std::is_same_v<Type, std::exception_ptr>) {
              if (arg) {
                std::rethrow_exception(arg);  // 重新抛出工作线程捕获的异常
              } else {
                // 理论上不应发生，但作为健壮性检查
                throw std::runtime_error(
                    "Worker thread sent null exception pointer.");
              }
            }
          },
          result);

      if (should_break) {
        break;  // 退出结果处理循环
      }
    }
  } catch (...) {
    // 捕获主线程中重新抛出的异常 (来自工作线程或回调函数)
    // 确保即使出错也要尝试 join 线程
    if (generation_thread.joinable()) {
      generation_thread.join();
    }
    throw;  // 将异常继续向外层传递
  }

  // --- Cleanup ---
  // 确保线程在函数正常结束时也被 join
  if (generation_thread.joinable()) {
    generation_thread.join();
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
