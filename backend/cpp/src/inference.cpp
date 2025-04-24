
#include "inference.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "base_model.hpp"
#include "common.hpp"
#include "cudaOP.cuh"
#include "llama.hpp"
#include "operators.hpp"
#include "qwen.hpp"
#define checkCudaErrors(call)                                           \
  do {                                                                  \
    cudaError_t err = call;                                             \
    if (err != cudaSuccess) {                                           \
      fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(err));                                 \
      throw std::runtime_error(cudaGetErrorString(err));                \
    }                                                                   \
  } while (0)
// 定义用于结果队列的类型
enum class Signal { EndOfStream };  // 定义 Signal 枚举
using GenerationResult =
    std::variant<uint32_t, Signal, std::exception_ptr>;  // 定义类型别名
// using GenerationResult = uint32_t*;
namespace py = pybind11;

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
      device_(device),
      d_states(nullptr) {  // 初始化 d_states 为 nullptr

  // std::cout
  //     << "[InferenceEngine::InferenceEngine] 初始化 InferenceEngine, device="
  //     << (device_ == Device::CUDA ? "CUDA" : "CPU") << std::endl;
  if (device_ == Device::CUDA) {
    cudaMalloc(&d_states, sizeof(curandState));
    int seed = std::chrono::system_clock::now().time_since_epoch().count();
    cuda_OP::init_curand(d_states, seed, 0, nullptr);
    std::cout
        << "[InferenceEngine::InferenceEngine] Moving InferenceEngine to CUDA"
        << std::endl;
    this->cuda();
  }
}

template <typename T>
InferenceEngine<T>::~InferenceEngine() {
  // 释放 CUDA 资源
  if (d_states != nullptr && device_ == Device::CUDA) {
    // 确保在释放前同步所有 CUDA 操作
    cudaDeviceSynchronize();
    cudaFree(d_states);
    d_states = nullptr;
  }
}

template <typename T>
uint32_t* InferenceEngine<T>::generate_next_token(ThreadPool& thread_pool,
                                                  uint32_t* input_ids,
                                                  float temperature,
                                                  float top_p, size_t top_k) {
  // std::cout << "[InferenceEngine::generate_next_token] 开始生成下一个 token"
  //           << std::endl;
  // 构造输入张量，取 input_ids 中最后一个 token, 放置在正确的设备上
  Tensor<uint32_t> input(input_ids, {1}, device_);
  // std::cout << "[InferenceEngine::generate_next_token] 输入 token: "
  //           << input_ids.back() << std::endl;
  // 更新 KV 缓存长度（为新 token 分配缓存空间）
  try {
    kv_cache_.resize(kv_cache_.size() + 1);
  } catch (const std::runtime_error& e) {
    std::cerr << "Error resizing KV cache: " << e.what() << std::endl;
    throw;
  }

  uint32_t* next_token;
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
  bind_this_thread_to_core(30);
  std::thread generation_thread([&, this, input_ids_copy = input_ids]() {
    try {
      using duration_unit = std::chrono::microseconds;
      const char* time_unit_str = "us"; // 用于打印

      // --- 初始化累加器 ---
      long long total_step_gpu_compute_us = 0;
      long long total_d2h_sync_us = 0;
      long long total_push_us = 0;
      long long total_loop_overhead_us = 0; // 估算循环本身开销
      bind_this_thread_to_core(31);
      uint32_t* next_token_gpu_ptr; // 指向 GPU 上的 next_token
      uint32_t next_token_host = -1; // CPU 上的 next_token 副本

      size_t input_size = input_ids_copy.size();

      auto prefill_start = std::chrono::high_resolution_clock::now();
      // --- Prefill ---
      { // Prefill scope
          kv_cache_.resize(kv_cache_.size() + input_size); // 调整大小移到 prefill 前
          std::vector<uint32_t> prefill_input = input_ids_copy;
          Tensor<uint32_t> input_tensor(std::move(prefill_input), {input_size}, this->device_);
          next_token_gpu_ptr = this->model_->prefill(
              &input_tensor, this->thread_pool_, &this->kv_cache_, top_k,
              temperature, top_p, this->d_states);
      }
      auto prefill_end = std::chrono::high_resolution_clock::now();
      auto prefill_elapsed_ms =
          std::chrono::duration_cast<std::chrono::milliseconds>(prefill_end - prefill_start)
              .count();

      // --- 处理 Prefill 的第一个 Token ---
      // 显式同步等待 Prefill 的 GPU 计算完成 (如果 prefill 是异步的)
      // 如果 prefill 函数内部保证同步返回，则不需要这句
      checkCudaErrors(cudaDeviceSynchronize()); // 确保 prefill 的 GPU 操作完成

      auto d2h_prefill_start = std::chrono::high_resolution_clock::now();
      // 从 GPU 获取 prefill 产生的第一个 token
      // 假设 prefill 返回的是设备指针，需要在默认流上拷贝回来
      checkCudaErrors(cudaMemcpyAsync(&next_token_host, next_token_gpu_ptr, sizeof(uint32_t),
                                       cudaMemcpyDeviceToHost, cudaStreamDefault));
      // 为了计时准确，同步等待 D2H 完成 (!!! 仅用于计时 !!!)
      checkCudaErrors(cudaStreamSynchronize(cudaStreamDefault));
      auto d2h_prefill_end = std::chrono::high_resolution_clock::now();
      auto d2h_prefill_us = std::chrono::duration_cast<duration_unit>(d2h_prefill_end - d2h_prefill_start).count();


      if (next_token_host == this->model_->get_eos_token_id()) {
        result_queue.push(Signal::EndOfStream);
        std::cout << "\n[Prefill Only Details]\n";
        std::cout << "  Prefill GPU + Sync Time (ms): " << prefill_elapsed_ms << "\n";
        std::cout << "  Prefill D2H + Sync Time (" << time_unit_str << "): " << d2h_prefill_us << "\n";
        std::cout << "[decode 耗时] 0 " << time_unit_str << std::endl << std::endl;
        return;
      }

      result_queue.push(next_token_host);

      size_t current_total_length = input_size + 1;
      uint32_t* last_token_gpu_ptr = next_token_gpu_ptr; // 下一轮的输入是 GPU 指针

      auto decode_start = std::chrono::high_resolution_clock::now();
      auto last_step_end_time = decode_start; // 用于计算循环开销

      while (current_total_length < max_length) {
        auto loop_overhead_start = std::chrono::high_resolution_clock::now();
        total_loop_overhead_us += std::chrono::duration_cast<duration_unit>(loop_overhead_start - last_step_end_time).count();

        auto step_gpu_compute_start = std::chrono::high_resolution_clock::now();
        // --- 生成下一个 token (主要是 GPU 计算) ---
        next_token_gpu_ptr = this->generate_next_token(this->thread_pool_, last_token_gpu_ptr,
                                                temperature, top_p, top_k);
        // 显式同步，确保 generate_next_token 的 GPU 计算完成 (!!! 仅用于计时 !!!)
        // 假设 generate_next_token 在其使用的流上完成所有工作
        // 如果它内部使用了特定流 stream_x, 则同步 cudaStreamSynchronize(stream_x)
        checkCudaErrors(cudaDeviceSynchronize()); // 或者同步特定流
        auto step_gpu_compute_end = std::chrono::high_resolution_clock::now();
        total_step_gpu_compute_us += std::chrono::duration_cast<duration_unit>(step_gpu_compute_end - step_gpu_compute_start).count();


        auto step_d2h_sync_start = std::chrono::high_resolution_clock::now();
        // --- 异步拷贝结果回 CPU ---
        checkCudaErrors(cudaMemcpyAsync(&next_token_host, next_token_gpu_ptr, sizeof(uint32_t),
                                        cudaMemcpyDeviceToHost, cudaStreamDefault)); // 假设用默认流
        // 为了计时准确，同步等待 D2H 完成 (!!! 仅用于计时 !!!)
        checkCudaErrors(cudaStreamSynchronize(cudaStreamDefault));
        auto step_d2h_sync_end = std::chrono::high_resolution_clock::now();
        total_d2h_sync_us += std::chrono::duration_cast<duration_unit>(step_d2h_sync_end - step_d2h_sync_start).count();


        auto step_logic_push_start = std::chrono::high_resolution_clock::now();
        // --- CPU 逻辑和 Push ---
        last_token_gpu_ptr = next_token_gpu_ptr; // 更新下一轮的输入指针
        current_total_length++;
        bool is_eos = (next_token_host == this->model_->get_eos_token_id());

        if (is_eos) {
          // 在 push 前记录结束时间
          auto push_start = std::chrono::high_resolution_clock::now();
          result_queue.push(Signal::EndOfStream);
          auto push_end = std::chrono::high_resolution_clock::now();
          total_push_us += std::chrono::duration_cast<duration_unit>(push_end - push_start).count();
          last_step_end_time = push_end; // 更新最后结束时间点
          break; // 跳出循环
        }

        auto push_start = std::chrono::high_resolution_clock::now();
        result_queue.push(next_token_host);
        auto push_end = std::chrono::high_resolution_clock::now();
        total_push_us += std::chrono::duration_cast<duration_unit>(push_end - push_start).count();
        last_step_end_time = push_end; // 更新最后结束时间点
      } // end while loop

      auto decode_end = std::chrono::high_resolution_clock::now();
      auto decode_elapsed_us = std::chrono::duration_cast<duration_unit>(decode_end - decode_start).count();
      auto post_loop_us = std::chrono::duration_cast<duration_unit>(decode_end - last_step_end_time).count();

      // --- 打印详细计时信息 ---
      std::cout << "\n[Detailed Timing (" << time_unit_str << ")]\n";
      std::cout << "  Prefill GPU + Sync Time (ms):       " << prefill_elapsed_ms << "\n"; // Prefill 仍用 ms
      std::cout << "  Prefill D2H + Sync Time:            " << d2h_prefill_us << "\n";
      std::cout << "  Total Decode Time (Outer Loop):     " << decode_elapsed_us << "\n";
      std::cout << "  --- Inside Decode Loop Breakdown ---\n";
      std::cout << "  Accumulated GPU Compute + Sync Time:" << total_step_gpu_compute_us << "\n";
      std::cout << "  Accumulated D2H Copy + Sync Time:   " << total_d2h_sync_us << "\n";
      std::cout << "  Accumulated Push to Queue Time:     " << total_push_us << "\n";
      std::cout << "  Accumulated Loop Overhead Est.:     " << total_loop_overhead_us << "\n"; // 循环本身开销+CPU逻辑
      std::cout << "  Sum of Loop Components:             "
                << (total_step_gpu_compute_us + total_d2h_sync_us + total_push_us + total_loop_overhead_us) << "\n";
      std::cout << "  Post-Loop Overhead:                 " << post_loop_us << "\n"; // 循环结束后到 decode_end 的时间
      std::cout << "  (Loop Comp. + Post-Loop Overhead):  "
                << (total_step_gpu_compute_us + total_d2h_sync_us + total_push_us + total_loop_overhead_us + post_loop_us) << "\n";
      std::cout << "  Discrepancy (Outer - Inner Sum):    "
                << (decode_elapsed_us - (total_step_gpu_compute_us + total_d2h_sync_us + total_push_us + total_loop_overhead_us + post_loop_us)) << "\n";
      std::cout << std::endl;
      result_queue.push(Signal::EndOfStream);

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
