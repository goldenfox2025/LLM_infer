#include "inference.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <chrono>
#include <iomanip>

#include "base_model.hpp"
#include "common.hpp"
#include "cudaOP.cuh"
#include "llama.hpp"
#include "operators.hpp"
#include "qwen.hpp"

// 关于如何测试Qwen2.5的图推理
// 1. 修改 `qwen.hpp` 中的 `use_cuda_graph_` 值
// 2. 运行 `./build.sh` 重新编译
// 3. 运行 `python frontend/chat.py --model_type qwen_bf16`
// 4. 观察模型信息中的 "CUDA Graph" 状态
// 5. 测试推理性能和正确性

// 使用 common.hpp 中定义的 checkCudaErrors 宏
// 定义用于结果队列的类型
enum class Signal { EndOfStream };                                            // 定义 Signal 枚举
using GenerationResult = std::variant<uint32_t, Signal, std::exception_ptr>;  // 定义类型别名
// using GenerationResult = uint32_t*;
namespace py = pybind11;

template <typename T>
KVCache<T>::KVCache(size_t n_layers, size_t max_seq_len, size_t head_dim, Device device, size_t initial_size)
    : n_layers_(n_layers), max_seq_len_(max_seq_len), head_dim_(head_dim), current_len_(0), device_(device) {
    // 初始化 KVCache

    // 分配连续内存，形状为 [n_layers, max_seq_len, head_dim]
    // 关键修复：强制KV Cache从独立内存分配，不使用prefill buffer
    // 通过设置is_prefill=false确保不会从prefill buffer分配
    k_cache_contiguous_ = Tensor<T>({n_layers_, max_seq_len_, head_dim_}, device_, false);
    v_cache_contiguous_ = Tensor<T>({n_layers_, max_seq_len_, head_dim_}, device_, false);

    // 分配一维 vector 存储所有 slice
    k_cache_slices_.resize(n_layers_ * max_seq_len_);
    v_cache_slices_.resize(n_layers_ * max_seq_len_);
    for (size_t layer = 0; layer < n_layers_; layer++) {
        for (size_t pos = 0; pos < max_seq_len_; pos++) {
            size_t idx = layer * max_seq_len_ + pos;
            // 利用 slice 方法构造对应的 view，
            // 对于 k_cache_contiguous_，取范围 [layer, pos, 0] 到 [layer+1, pos+1, head_dim_]
            k_cache_slices_[idx] = k_cache_contiguous_.slice({layer, pos, 0}, {layer + 1, pos + 1, head_dim_});
            v_cache_slices_[idx] = v_cache_contiguous_.slice({layer, pos, 0}, {layer + 1, pos + 1, head_dim_});
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
    if (device_ == Device::CUDA)
        return *this;

    device_ = Device::CUDA;
    // 将连续内存移动到 CUDA 设备
    k_cache_contiguous_ = k_cache_contiguous_.cuda();
    v_cache_contiguous_ = v_cache_contiguous_.cuda();

    // 重新构造所有 slice（由于连续内存的 data pointer 更新）
    for (size_t layer = 0; layer < n_layers_; layer++) {
        for (size_t pos = 0; pos < max_seq_len_; pos++) {
            size_t idx = layer * max_seq_len_ + pos;
            k_cache_slices_[idx] = k_cache_contiguous_.slice({layer, pos, 0}, {layer + 1, pos + 1, head_dim_});
            v_cache_slices_[idx] = v_cache_contiguous_.slice({layer, pos, 0}, {layer + 1, pos + 1, head_dim_});
        }
    }
    return *this;
}

template <typename T>
KVCache<T>& KVCache<T>::cpu() {
    if (device_ == Device::CPU)
        return *this;

    device_ = Device::CPU;
    // 将连续内存移回 CPU
    k_cache_contiguous_ = k_cache_contiguous_.cpu();
    v_cache_contiguous_ = v_cache_contiguous_.cpu();

    // 重新构造所有 slice
    for (size_t layer = 0; layer < n_layers_; layer++) {
        for (size_t pos = 0; pos < max_seq_len_; pos++) {
            size_t idx = layer * max_seq_len_ + pos;
            k_cache_slices_[idx] = k_cache_contiguous_.slice({layer, pos, 0}, {layer + 1, pos + 1, head_dim_});
            v_cache_slices_[idx] = v_cache_contiguous_.slice({layer, pos, 0}, {layer + 1, pos + 1, head_dim_});
        }
    }
    return *this;
}

template <typename T>
std::pair<const Tensor<T>, const Tensor<T>> KVCache<T>::get_contiguous_tensor(size_t layer) const {
    Tensor<T> K = k_cache_contiguous_.slice({layer, 0, 0}, {layer + 1, current_len_, head_dim_}).squeeze(0);
    Tensor<T> V = v_cache_contiguous_.slice({layer, 0, 0}, {layer + 1, current_len_, head_dim_}).squeeze(0);

    return {K, V};
}

template <typename T>
std::pair<Tensor<T>, Tensor<T>> KVCache<T>::get_layer_view(size_t layer) {
    if (layer >= n_layers_) {
        throw std::runtime_error("KVCache: Layer index out of range in get_layer_view");
    }
    // Return a view (slice) of the specified layer.
    // The view is writable and points to the continuous memory block of that layer.
    Tensor<T> k_view = k_cache_contiguous_.slice({layer, 0, 0}, {layer + 1, max_seq_len_, head_dim_}).squeeze(0);
    Tensor<T> v_view = v_cache_contiguous_.slice({layer, 0, 0}, {layer + 1, max_seq_len_, head_dim_}).squeeze(0);
    return {k_view, v_view};
}

// 显式实例化模板类
template class KVCache<float>;
template class KVCache<__nv_bfloat16>;

// ------------------------
// InferenceEngine 实现
// ------------------------

// 初始化静态变量
template <typename T>
bool InferenceEngine<T>::has_warmed_up_ = false;
template <typename T>
InferenceEngine<T>::InferenceEngine(std::shared_ptr<BaseModel> model, Device device)
    : model_(model),
      // 使用模型参数初始化 KVCache，并指定设备
      kv_cache_(model_->get_n_layers(), model_->get_max_seq_len(), model_->get_head_dim() * model_->get_n_kv_heads(),
                device),
      thread_pool_(4),
      device_(device),
      d_states(nullptr) {  // 初始化 d_states 为 nullptr

    // 如果请求的设备是CUDA，确保模型也在CUDA上
    if (device_ == Device::CUDA) {
        // 检查模型是否已经在CUDA上，如果不是则移动到CUDA
        if (model_->device() != Device::CUDA) {
            model_->cuda();
        }

        // 初始化CUDA资源
        cudaError_t err = cudaMalloc(&d_states, sizeof(curandState));
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to allocate CUDA memory for curand states: " +
                                     std::string(cudaGetErrorString(err)));
        }
        int seed = std::chrono::system_clock::now().time_since_epoch().count();
        cuda_OP::init_curand(d_states, seed, 0, nullptr);
        this->cuda();
    }

    // 在设备处理之后打印最终设备状态
    Device model_device = model_->device();
    std::cout << "【引擎初始化完成】模型当前设备: " << (model_device == Device::CUDA ? "CUDA" : "CPU") << std::endl;
    std::cout << "【引擎初始化完成】引擎设备: " << (device_ == Device::CUDA ? "CUDA" : "CPU") << std::endl;
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
uint32_t* InferenceEngine<T>::generate_next_token(ThreadPool& thread_pool, uint32_t* input_ids, float temperature,
                                                  float top_p, size_t top_k) {
    // 创建GPU计时器
    GpuTimer token_gen_timer;
    token_gen_timer.start();

    // 修复：input_ids是GPU指针，需要用特殊构造函数
    // 构造输入张量，取 input_ids 中最后一个 token, 放置在正确的设备上
    Tensor<uint32_t> input(input_ids, {1}, device_);  // 这里使用GPU指针构造函数

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
        next_token = model_->forward(&input, thread_pool, &kv_cache_, top_k, temperature, top_p, d_states);
    } else {
        next_token = model_->forward(&input, thread_pool, &kv_cache_, top_k, temperature, top_p);
    }

    // 停止计时
    token_gen_timer.stop();

    // 输出生成时间（可选，取消注释以启用）
    // std::cout << "Token生成耗时: " << std::fixed << std::setprecision(2)
    //           << token_gen_timer.milliseconds() << " 毫秒" << std::endl
    //           << std::flush;

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
#include <numeric>  // 用于 std::accumulate
#include <vector>   // 用于 std::vector

template <typename T>
void InferenceEngine<T>::generate_with_callback(const std::vector<uint32_t>& input_ids, size_t max_length,
                                                float temperature, float top_p, size_t top_k,
                                                std::function<void(uint32_t)> callback) {
    // 如果是第一次调用，执行预热
    if (!has_warmed_up_ && device_ == Device::CUDA) {
        std::cout << "执行CUDA预热..." << std::endl << std::flush;

        // 创建一个小的输入序列用于预热
        std::vector<uint32_t> warmup_input(64, 1);

        // 保存当前KV缓存状态
        size_t original_kv_size = kv_cache_.size();

        // 执行预热操作
        try {
            // 创建GPU计时器
            GpuTimer warmup_timer;
            warmup_timer.start();

            // 设置prefill阶段标志
            GlobalCudaMemoryPool::set_prefill_phase(true);

            // 调整KV缓存大小
            kv_cache_.resize(warmup_input.size());

            // 创建输入张量 - 使用拷贝构造而不是移动构造
            std::vector<uint32_t> warmup_input_copy = warmup_input;
            Tensor<uint32_t> input_tensor(std::move(warmup_input_copy), {warmup_input.size()}, device_);

            // 执行prefill操作
            uint32_t* warmup_token =
                model_->prefill(&input_tensor, thread_pool_, &kv_cache_, top_k, temperature, top_p, d_states);
            GlobalCudaMemoryPool::set_prefill_phase(false);
            // 如果是QwenModel且支持CUDA图，在预热阶段调用一次forward来初始化CUDA图
            // 这样可以使用真实的KV cache
            std::cout << "检查是否需要初始化CUDA图..." << std::endl;

            // 尝试向下转型为QwenModel，支持两种模板实例化
            auto qwen_model_bf16 = dynamic_cast<QwenModel<__nv_bfloat16>*>(model_.get());
            auto qwen_model_float = dynamic_cast<QwenModel<float>*>(model_.get());

            if (qwen_model_bf16 || qwen_model_float) {
                std::cout << "检测到QwenModel，开始CUDA图初始化..." << std::endl;
                try {
                    // 调整KV缓存大小为单token推理
                    kv_cache_.resize(1);

                    // 创建单token输入用于图初始化
                    std::vector<uint32_t> graph_init_input = {9707};  // 包含token 9707的向量
                    Tensor<uint32_t> graph_input_tensor_(std::move(graph_init_input), {1}, device_);

                    // 调用forward来触发CUDA图初始化
                    uint32_t* graph_warmup_token = model_->forward(&graph_input_tensor_, thread_pool_, &kv_cache_,
                                                                   top_k, temperature, top_p, d_states);

                    std::cout << "CUDA图初始化完成！" << std::endl;
                } catch (const std::exception& e) {
                    std::cout << "CUDA图初始化跳过或失败: " << e.what() << std::endl;
                    // 不是致命错误，继续执行
                }
            }
            // 停止计时
            warmup_timer.stop();

            // 重置KV缓存
            kv_cache_.clear();

            // 重置prefill buffer
            GlobalCudaMemoryPool::reset_prefill_buffer();

            // 标记已完成预热
            has_warmed_up_ = true;
            kv_cache_.resize(0);
            std::cout << "CUDA预热完成，耗时: " << std::fixed << std::setprecision(2) << warmup_timer.milliseconds()
                      << " 毫秒" << std::endl
                      << std::flush;
        } catch (const std::exception& e) {
            std::cerr << "预热过程中发生错误: " << e.what() << std::endl;
            // 即使预热失败，也继续执行正常的推理
        }
    }

    ThreadSafeQueue<GenerationResult> result_queue;
    bind_this_thread_to_core(3);
    std::thread generation_thread([&, this, input_ids_copy = input_ids]() {
        try {
            uint32_t* next_token_gpu_ptr;   // 指向 GPU 上的 next_token
            uint32_t next_token_host = -1;  // CPU 上的 next_token 副本
            size_t input_size = input_ids_copy.size();

            // 声明计时器，确保在整个函数范围内可见
            GpuTimer total_prefill_timer;
            total_prefill_timer.start();

            {
                // 设置prefill阶段标志，启用prefill模式
                GlobalCudaMemoryPool::set_prefill_phase(true);
                std::cerr << "进入prefill阶段，序列长度: " << input_size << std::endl;

                // 开始计时 - 仅计算部分
                GpuTimer prefill_timer;
                prefill_timer.start();

                kv_cache_.resize(kv_cache_.size() + input_size);  // 调整大小移到 prefill 前
                std::vector<uint32_t> prefill_input = input_ids_copy;
                Tensor<uint32_t> input_tensor(std::move(prefill_input), {input_size}, this->device_);
                next_token_gpu_ptr = this->model_->prefill(&input_tensor, this->thread_pool_, &this->kv_cache_, top_k,
                                                           temperature, top_p, this->d_states);

                // 确保所有CUDA操作完成后再停止计时
                prefill_timer.stop();

                // 关闭prefill阶段标志
                GlobalCudaMemoryPool::set_prefill_phase(false);

                // 输出计时结果，确保立即刷新输出缓冲区
                std::cout << "Prefill阶段完成，耗时: " << std::fixed << std::setprecision(2)
                          << prefill_timer.milliseconds() << " 毫秒" << std::endl
                          << std::flush;

                std::cerr << "退出prefill阶段" << std::endl;
            }

            // --- 处理 Prefill 的第一个 Token ---
            // 开始计时 - 第一个token处理
            GpuTimer token_timer;
            token_timer.start();

            // 从 GPU 获取 prefill 产生的第一个 token
            checkCudaErrors(cudaMemcpyAsync(&next_token_host, next_token_gpu_ptr, sizeof(uint32_t),
                                            cudaMemcpyDeviceToHost, cudaStreamDefault));
            checkCudaErrors(cudaStreamSynchronize(cudaStreamDefault));

            // 结束计时 - 第一个token处理
            token_timer.stop();

            // 输出第一个token处理时间
            std::cout << "第一个Token处理耗时: " << std::fixed << std::setprecision(2) << token_timer.milliseconds()
                      << " 毫秒" << std::endl
                      << std::flush;

            // 计算并输出总prefill时间（包括计算和第一个token处理）
            total_prefill_timer.stop();
            std::cout << "总Prefill过程耗时: " << std::fixed << std::setprecision(2)
                      << total_prefill_timer.milliseconds() << " 毫秒" << std::endl
                      << std::flush;

            if (next_token_host == this->model_->get_eos_token_id()) {
                result_queue.push(Signal::EndOfStream);
                return;
            }

            result_queue.push(next_token_host);

            size_t current_total_length = input_size + 1;
            uint32_t* last_token_gpu_ptr = next_token_gpu_ptr;  // 下一轮的输入是 GPU 指针

            // ==================== 新增代码段：初始化解码计时器 ====================
            std::vector<float> decode_times;
            // ===============================================================

            while (current_total_length < max_length) {
                // 创建GPU计时器用于整个token生成和处理过程
                GpuTimer full_token_timer;
                full_token_timer.start();

                // --- 生成下一个 token (主要是 GPU 计算) ---
                next_token_gpu_ptr =
                    this->generate_next_token(this->thread_pool_, last_token_gpu_ptr, temperature, top_p, top_k);

                // --- 异步拷贝结果回 CPU ---
                checkCudaErrors(
                    cudaMemcpyAsync(&next_token_host, next_token_gpu_ptr, sizeof(uint32_t), cudaMemcpyDeviceToHost));

                // 等待GPU计算和数据传输完成，以确保计时准确
                checkCudaErrors(cudaStreamSynchronize(cudaStreamDefault));

                // 停止整体计时
                full_token_timer.stop();

                // ==================== 新增代码段：记录本次解码时间 ====================
                decode_times.push_back(full_token_timer.milliseconds());
                // =================================================================

                // --- CPU 逻辑和 Push ---
                last_token_gpu_ptr = next_token_gpu_ptr;  // 更新下一轮的输入指针
                current_total_length++;
                bool is_eos = (next_token_host == this->model_->get_eos_token_id());

                if (is_eos) {
                    result_queue.push(Signal::EndOfStream);
                    break;  // 跳出循环
                }

                result_queue.push(next_token_host);
            }  // end while loop

            result_queue.push(Signal::EndOfStream);

            // ==================== 新增代码段：计算并打印解码性能统计 ====================
            if (!decode_times.empty()) {
                float total_decode_time = std::accumulate(decode_times.begin(), decode_times.end(), 0.0f);
                float average_decode_time = total_decode_time / decode_times.size();

                std::cout << "\n-------------------- 解码性能统计 --------------------" << std::endl;
                std::cout << "解码Token数量: " << decode_times.size() << " 个" << std::endl;
                std::cout << std::fixed << std::setprecision(2) << "总解码时间: " << total_decode_time << " 毫秒"
                          << std::endl;
                std::cout << std::fixed << std::setprecision(2) << "平均解码时间: " << average_decode_time
                          << " 毫秒/Token" << std::endl;
                std::cout << std::fixed << std::setprecision(2) << "平均解码速率: " << (1000.0f / average_decode_time)
                          << " Token/秒" << std::endl;
                std::cout << "--------------------------------------------------------" << std::endl << std::flush;
            }
            // ========================================================================

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
                            std::cerr << "Python error in callback: " << e.what() << std::endl;
                            // 将 Python 异常包装成 C++ 异常并抛出
                            throw std::runtime_error("Python callback failed: " + std::string(e.what()));
                        } catch (const std::exception& e) {  // 捕获标准 C++ 异常
                            std::cerr << "C++ error in callback: " << e.what() << std::endl;
                            throw;  // 重新抛出
                        } catch (...) {
                            std::cerr << "Unknown error during callback execution." << std::endl;
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
                            throw std::runtime_error("Worker thread sent null exception pointer.");
                        }
                    }
                },
                result);

            if (should_break) {
                break;  // 退出结果处理循环
            }
        }
    } catch (...) {
        std::cerr << "No!" << std::endl;
        if (generation_thread.joinable()) {
            generation_thread.join();
        }
        throw;
    }

    // --- Cleanup ---
    // 确保线程在函数正常结束时也被 join
    if (generation_thread.joinable()) {
        generation_thread.join();
    }

    // 重置prefill buffer，但不释放内存，以便下次使用
    // 这样可以避免频繁的内存分配和释放，提高性能
    GlobalCudaMemoryPool::reset_prefill_buffer();
}
template <typename T>
void InferenceEngine<T>::reset() {
    kv_cache_.clear();

    // 重置模型的异步预准备状态
    // 尝试转换为QwenModel并调用重置方法
    auto qwen_model_bf16 = dynamic_cast<QwenModel<__nv_bfloat16>*>(model_.get());
    auto qwen_model_float = dynamic_cast<QwenModel<float>*>(model_.get());
}
template <typename T>
InferenceEngine<T>& InferenceEngine<T>::cuda() {
    if (device_ == Device::CUDA) {
        return *this;
    }
    if (model_->device() == Device::CPU) {
        model_->cuda();
    }

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
