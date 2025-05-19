#include "speculative_decoder.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cudaOP.cuh"
#include "operators/unified_operators.hpp"
#include "qwen3.hpp"
// 定义检查CUDA错误的宏
#define checkCudaErrors(call)                                                                           \
    do {                                                                                                \
        cudaError_t err = call;                                                                         \
        if (err != cudaSuccess) {                                                                       \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            throw std::runtime_error(cudaGetErrorString(err));                                          \
        }                                                                                               \
    } while (0)

// 构造函数实现
template <typename T>
SpeculativeDecoder<T>::SpeculativeDecoder(std::shared_ptr<BaseModel> target_model,
                                          std::shared_ptr<BaseModel> draft_model, size_t spec_length,
                                          size_t thread_count)
    : target_model_(target_model),
      draft_model_(draft_model),
      target_kv_cache_(target_model->get_n_layers(), target_model->get_max_seq_len(),
                       target_model->get_head_dim() * target_model->get_n_kv_heads(), Device::CUDA),
      draft_kv_cache_(draft_model->get_n_layers(), draft_model->get_max_seq_len(),
                      draft_model->get_head_dim() * draft_model->get_n_kv_heads(), Device::CUDA),
      thread_pool_(thread_count),  // 使用传入的线程数
      device_(Device::CUDA),
      spec_length_(spec_length),
      d_states(nullptr),
      d_reuse_token(nullptr),
      main_stream_(nullptr),
      draft_stream_(nullptr),
      verify_stream_(nullptr) {
    // 确保两个模型都在CUDA上
    if (target_model_->device() != Device::CUDA) {
        target_model_->cuda();
    }
    if (draft_model_->device() != Device::CUDA) {
        draft_model_->cuda();
    }
    draft_kv_cache_.clear();
    target_kv_cache_.clear();
    // 初始化CUDA资源
    init_cuda_resources();

    std::cout << "【投机解码器初始化完成】" << std::endl;
    std::cout << "目标模型: " << target_model_->get_n_layers() << " 层" << std::endl;
    std::cout << "草稿模型: " << draft_model_->get_n_layers() << " 层" << std::endl;
    std::cout << "投机长度: " << spec_length_ << std::endl;
    std::cout << "线程池大小: " << thread_count << std::endl;
}

// 析构函数实现
template <typename T>
SpeculativeDecoder<T>::~SpeculativeDecoder() {
    // 释放CUDA资源
    free_cuda_resources();
}

// 初始化CUDA资源
template <typename T>
void SpeculativeDecoder<T>::init_cuda_resources() {
    // 分配CUDA随机状态
    cudaMalloc(&d_states, sizeof(curandState));
    int seed = std::chrono::system_clock::now().time_since_epoch().count();
    cuda_OP::init_curand(d_states, seed, 0, nullptr);

    // 创建CUDA流，用于异步操作
    cudaStreamCreate(&main_stream_);
    cudaStreamCreate(&draft_stream_);
    cudaStreamCreate(&verify_stream_);

    // 使用内存池分配固定内存，避免频繁的分配和释放
    if (GlobalCudaMemoryPool::has_tag(kReuseTokenTag)) {
        d_reuse_token = static_cast<uint32_t*>(GlobalCudaMemoryPool::get_tagged_memory(kReuseTokenTag));
    } else {
        d_reuse_token = static_cast<uint32_t*>(GlobalCudaMemoryPool::allocate_tagged(kReuseTokenTag, sizeof(uint32_t)));
    }

    // 如果内存池分配失败，则使用普通的CUDA内存分配
    if (d_reuse_token == nullptr) {
        checkCudaErrors(cudaMalloc(&d_reuse_token, sizeof(uint32_t)));
    }
}

// 释放CUDA资源
template <typename T>
void SpeculativeDecoder<T>::free_cuda_resources() {
    // 同步所有流，确保所有操作完成
    if (main_stream_) {
        cudaStreamSynchronize(main_stream_);
        cudaStreamDestroy(main_stream_);
        main_stream_ = nullptr;
    }

    if (draft_stream_) {
        cudaStreamSynchronize(draft_stream_);
        cudaStreamDestroy(draft_stream_);
        draft_stream_ = nullptr;
    }

    if (verify_stream_) {
        cudaStreamSynchronize(verify_stream_);
        cudaStreamDestroy(verify_stream_);
        verify_stream_ = nullptr;
    }

    if (d_states != nullptr) {
        cudaDeviceSynchronize();
        cudaFree(d_states);
        d_states = nullptr;
    }

    // 不需要释放标记内存，由内存池管理
    if (d_reuse_token != nullptr && !GlobalCudaMemoryPool::has_tag(kReuseTokenTag)) {
        cudaFree(d_reuse_token);
        d_reuse_token = nullptr;
    }
}

// 使用草稿模型生成多个token，直接返回GPU指针数组
template <typename T>
std::vector<uint32_t*> SpeculativeDecoder<T>::generate_draft_tokens_gpu(uint32_t* input_token, size_t num_tokens,
                                                                        float temperature, float top_p, size_t top_k) {
    // 预分配GPU指针数组，避免频繁扩容
    std::vector<uint32_t*> gpu_tokens;
    gpu_tokens.push_back(input_token);
    int initial_kv_size = draft_kv_cache_.size();
    try {
        uint32_t* d_current_token = input_token;

        // 复制打印第一个token
        uint32_t first_token_value;
        cudaMemcpyAsync(&first_token_value, d_current_token, sizeof(uint32_t), cudaMemcpyDeviceToHost, draft_stream_);
        cudaStreamSynchronize(draft_stream_);
        std::cout << "【草稿模型】生成token[0]: " << first_token_value << std::endl;
        // 生成剩余token，使用forward进行自回归生成
        for (size_t i = 0; i < num_tokens; i++) {
            // 构造输入张量，直接使用GPU上的指针
            Tensor<uint32_t> input(d_current_token, {1}, device_);
            draft_kv_cache_.resize(draft_kv_cache_.size() + 1);
            std::cout << "【草稿模型】KV缓存大小: " << draft_kv_cache_.size() << std::endl;
            uint32_t* next_token =
                draft_model_->forward(&input, thread_pool_, &draft_kv_cache_, top_k, temperature, top_p, d_states);
            // 检查生成是否成功
            // 打印nexttoken地址
            std::cout << "【草稿模型】生成token[" << i + 1 << "]: " << next_token << std::endl;
            if (next_token == nullptr) {
                // 如果生成失败，需要将KV缓存大小调整回实际使用的大小
                draft_kv_cache_.resize(initial_kv_size + i);
                std::cout << "【草稿模型】生成失败，提前结束生成" << std::endl;
                break;
            }

            // 保存GPU指针，避免不必要的GPU-CPU数据传输
            gpu_tokens.push_back(next_token);

            // 更新当前token为新生成的token
            d_current_token = next_token;

            // 检查是否需要提前停止生成（例如EOS token）- 使用异步拷贝
            uint32_t token_value;
            cudaMemcpyAsync(&token_value, next_token, sizeof(uint32_t), cudaMemcpyDeviceToHost, draft_stream_);

            // 同步以获取token值用于打印
            cudaStreamSynchronize(draft_stream_);
            std::cout << "【草稿模型】生成token[" << i + 1 << "]: " << token_value << std::endl;

            // 只在循环的最后一次迭代或者需要检查EOS时同步
            if (i == num_tokens) {
                if (token_value == draft_model_->get_eos_token_id()) {
                    std::cout << "【草稿模型】生成EOS token，提前结束生成" << std::endl;
                    break;  // 如果是EOS token，提前结束生成
                }
            } else {
                // 对于中间的token，我们可以继续生成下一个token而不等待
                // 这里可以使用事件来检查前一个操作是否完成，但为了简化，我们暂时不做这个优化
            }
        }

        std::cout << "【草稿模型】本次共生成 " << gpu_tokens.size() << " 个token" << std::endl;

    } catch (const std::exception& e) {
        // 记录错误并返回已生成的token
        std::cout << "草稿模型生成token时出错: " << e.what() << std::endl;
    }

    return gpu_tokens;
}

// 批处理批量从GPU读取token值
template <typename T>
void SpeculativeDecoder<T>::generate_with_callback(const std::vector<uint32_t>& input_ids, size_t max_length,
                                                   float temperature, float top_p, size_t top_k,
                                                   std::function<void(uint32_t)> callback) {
    try {
        std::vector<uint32_t> current_ids;
        current_ids.reserve(max_length);  // 提前分配最大可能的大小
        current_ids = input_ids;

        // 确保输入序列非空
        if (current_ids.empty()) {
            std::cout << "错误: 输入序列为空" << std::endl;
            return;
        }
        uint32_t first_target_token_value = -1;
        // 1. 目标模型初始化 - 一次性预分配足够的GPU内存
        {
            // 为整个提示词序列预分配足够的内存，避免后续频繁调整大小
            // 进入prefill或forward时，需要手动扩展出当前输入的长度
            // 已有长度基础上，扩展输入长度
            target_kv_cache_.resize(target_kv_cache_.size() + current_ids.size());

            // 首次使用prefill处理整个提示词序列 - 直接创建张量而不是移动
            Tensor<uint32_t> input_tensor({current_ids.begin(), current_ids.end()}, {current_ids.size()}, device_);
            uint32_t* first_token = target_model_->prefill(&input_tensor, thread_pool_, &target_kv_cache_, top_k,
                                                           temperature, top_p, d_states);
            // // 再进行一次forward
            // target_kv_cache_.resize(target_kv_cache_.size() + 1);
            // uint32_t* first_token_forward = target_model_->forward(&input_tensor, thread_pool_, &target_kv_cache_,
            //                                                        top_k, temperature, top_p, d_states);
            if (first_token == nullptr) {
                std::cout << "警告: 目标模型初始预填充返回空指针" << std::endl;
                return;
            }

            // 从GPU读取第一个token值 - 使用异步拷贝

            cudaMemcpyAsync(&first_target_token_value, first_token, sizeof(uint32_t), cudaMemcpyDeviceToHost,
                            main_stream_);
            cudaStreamSynchronize(main_stream_);  // 确保拷贝完成

            // 回调函数处理第一个token
            callback(first_target_token_value);

            // 检查是否为EOS
            if (first_target_token_value == target_model_->get_eos_token_id()) {
                return;
            }
        }
        uint32_t first_draft_token_value = -1;
        // current_ids.push_back(first_target_token_value);
        // 2. 草稿模型初始化 - 同样预分配内存并使用批量处理
        {
            // 为草稿模型分配空间
            // 已经添加了第一个目标token
            draft_kv_cache_.resize(draft_kv_cache_.size() + current_ids.size());
            std::cout << "草稿模型初始化，KV缓存大小为: " << draft_kv_cache_.size() << std::endl;
            // 使用prefill为草稿模型准备KV缓存 - 直接传递数据不创建副本
            Tensor<uint32_t> draft_input_tensor({current_ids.begin(), current_ids.end()}, {current_ids.size()},
                                                device_);
            uint32_t* draft_first_token = draft_model_->prefill(&draft_input_tensor, thread_pool_, &draft_kv_cache_,
                                                                top_k, temperature, top_p, d_states);

            // 从GPU读取第一个token值 - 使用异步拷贝
            cudaMemcpyAsync(&first_draft_token_value, draft_first_token, sizeof(uint32_t), cudaMemcpyDeviceToHost,
                            main_stream_);
            cudaStreamSynchronize(main_stream_);  // 确保拷贝完成
        }
        current_ids.push_back(first_target_token_value);
        // current_ids.pop_back();
        std::cout << "草稿模型初始化完成，第一个token值为: " << first_draft_token_value << std::endl;
        // 3. 不需要额外分配GPU内存，直接使用预分配的重用内存
        // 将最大迭代次数限制为剩余的生成长度
        size_t max_iterations = max_length - current_ids.size();
        size_t iteration = 0;

        // 主循环：投机解码
        while (current_ids.size() < max_length && iteration < max_iterations) {
            try {
                // 获得目标模型最后推理出的token
                const uint32_t last_token = current_ids.back();
                cudaMemcpyAsync(d_reuse_token, &last_token, sizeof(uint32_t), cudaMemcpyHostToDevice, main_stream_);
                cudaStreamSynchronize(main_stream_);  // 确保拷贝完成
                std::vector<uint32_t*> gpu_tokens;
                gpu_tokens = generate_draft_tokens_gpu(d_reuse_token, spec_length_, temperature, top_p, top_k);

                // 验证草稿模型生成的token
                // 直接使用GPU指针进行验证，避免不必要的GPU-CPU数据传输
                std::vector<uint32_t> verified_tokens;
                verified_tokens.reserve(gpu_tokens.size());

                // 使用新的GPU版本的验证函数
                size_t match_length = verify_draft_tokens_gpu(current_ids, gpu_tokens, temperature, top_p, top_k,
                                                              verified_tokens, main_stream_);

                // verified_tokens一定会包含至少一个token(要么是匹配的token，要么是目标模型生成的替代token)

                // 批量添加验证通过的token - 避免逐个添加的开销
                bool found_eos = false;

                // 检查是否有EOS token并找到位置
                size_t eos_pos = verified_tokens.size();
                std::cout << "【验证】验证通过的token数量: " << verified_tokens.size() << std::endl;
                for (size_t i = 0; i < verified_tokens.size(); i++) {
                    if (verified_tokens[i] == target_model_->get_eos_token_id()) {
                        eos_pos = i;
                        found_eos = true;
                        break;
                    }
                }

                // 只添加到EOS标记处（如果有）
                size_t tokens_to_add = found_eos ? eos_pos + 1 : verified_tokens.size();
                current_ids.insert(current_ids.end(), verified_tokens.begin(), verified_tokens.begin() + tokens_to_add);
                std::cout << "【验证】当前输入序列: ";
                for (size_t i = 0; i < current_ids.size(); i++) {
                    std::cout << current_ids[i] << " ";
                }
                std::cout << std::endl;
                // 批量调用回调函数
                for (size_t i = 0; i < tokens_to_add; i++) {
                    callback(verified_tokens[i]);
                }

                // 如果找到了EOS，提前结束生成
                if (found_eos) {
                    break;
                }

                iteration++;
            } catch (const std::exception& e) {
                std::cout << "投机解码迭代中出错: " << e.what() << std::endl;

                // 尝试使用标准解码继续
                try {
                    if (!current_ids.empty()) {
                        uint32_t last_token = current_ids.back();
                        cudaMemcpyAsync(d_reuse_token, &last_token, sizeof(uint32_t), cudaMemcpyHostToDevice,
                                        main_stream_);
                        cudaStreamSynchronize(main_stream_);

                        Tensor<uint32_t> input(d_reuse_token, {1}, device_);

                        // 调整KV缓存大小
                        target_kv_cache_.resize(target_kv_cache_.size() + 1);

                        uint32_t* next_token = target_model_->forward(&input, thread_pool_, &target_kv_cache_, top_k,
                                                                      temperature, top_p, d_states);

                        if (next_token != nullptr) {
                            uint32_t token_value;
                            cudaMemcpyAsync(&token_value, next_token, sizeof(uint32_t), cudaMemcpyDeviceToHost,
                                            main_stream_);
                            cudaStreamSynchronize(main_stream_);

                            current_ids.push_back(token_value);
                            callback(token_value);

                            // 检查是否生成了EOS token
                            if (token_value == target_model_->get_eos_token_id()) {
                                break;
                            }
                        } else {
                            std::cout << "错误恢复失败，结束生成" << std::endl;
                            break;
                        }
                    }
                } catch (const std::exception& e) {
                    std::cout << "错误恢复过程中出错: " << e.what() << std::endl;
                    break;
                }

                iteration++;
            }
        }

    } catch (const std::exception& e) {
        std::cout << "投机解码过程中出错: " << e.what() << std::endl;
    }
}

// 批量验证草稿模型生成的token - GPU指针版本
template <typename T>
size_t SpeculativeDecoder<T>::verify_draft_tokens_gpu(const std::vector<uint32_t>& prefix_tokens,
                                                      const std::vector<uint32_t*>& draft_tokens_gpu, float temperature,
                                                      float top_p, size_t top_k, std::vector<uint32_t>& verified_tokens,
                                                      cudaStream_t stream) {
    // 记录最长匹配长度
    size_t max_match_length = 0;
    verified_tokens.clear();

    // 检查输入有效性
    if (draft_tokens_gpu.empty() || draft_tokens_gpu.size() == 1) {
        std::cout << "【验证】草稿模型生成的token数量为0或1，直接返回" << std::endl;
        return 0;  // 如果没有token或只有一个token（输入token），直接返回
    }
    auto qwen3 = std::dynamic_pointer_cast<Qwen3Model<T>>(target_model_);
    try {
        // 使用验证专用流
        cudaStream_t verify_stream = verify_stream_;

        // 完整前缀序列 - 使用现有的向量并预分配足够空间避免频繁重新分配
        std::vector<uint32_t> input_sequence;

        input_sequence = prefix_tokens;

        // 保存目标模型当前的KV缓存大小
        size_t original_cache_size = target_kv_cache_.size();
        std::cout << "【验证】目标模型当前KV缓存大小: " << original_cache_size << std::endl;
        // 预先扩展KV缓存到最大可能的大小，避免频繁调整
        target_kv_cache_.resize(original_cache_size + 1);
        std::cout << "【验证】为下轮推理扩展KV缓存: " << target_kv_cache_.size() << std::endl;

        // 记录最后一个目标模型生成的token
        uint32_t last_target_token_value = 0;
        int found_mismatch = -1;

        std::cout << "【验证开始】准备验证草稿模型生成的 " << draft_tokens_gpu.size() - 1 << " 个token" << std::endl;

        // 使用prefill处理整个前缀序列
        // Tensor<uint32_t> input_tensor({prefix_tokens.begin(), prefix_tokens.end()}, {prefix_tokens.size()}, device_);

        Tensor<uint32_t> input_tensor(draft_tokens_gpu[0], {1}, device_);
        uint32_t* target_token =
            target_model_->forward(&input_tensor, thread_pool_, &target_kv_cache_, top_k, temperature, top_p, d_states);

        // 检查prefill是否成功
        if (target_token == nullptr) {
            std::cout << "【验证错误】目标模型prefill返回空指针" << std::endl;
            target_kv_cache_.resize(original_cache_size);  // 恢复原始大小
            return 0;
        }

        // 获取第一个token值 - 使用异步拷贝
        cudaMemcpyAsync(&last_target_token_value, target_token, sizeof(uint32_t), cudaMemcpyDeviceToHost,
                        verify_stream);
        cudaStreamSynchronize(verify_stream);  // 确保拷贝完成，这里必须同步

        std::cout << "【目标模型】prefill生成第一个token: " << last_target_token_value << std::endl;

        // 现在使用forward逐个验证草稿token
        size_t current_cache_size = prefix_tokens.size();
        uint32_t* last_target_token;
        // 从第0个token开始验证所有token
        for (size_t i = 0; i < draft_tokens_gpu.size() - 1; i++) {
            std::cout << "【验证】验证草稿模型token地址: " << draft_tokens_gpu[i + 1] << std::endl;
            // 直接从GPU获取token值，避免不必要的CPU复制
            uint32_t draft_token_value;
            cudaMemcpyAsync(&draft_token_value, draft_tokens_gpu[i + 1], sizeof(uint32_t), cudaMemcpyDeviceToHost,
                            verify_stream);

            // 这里必须同步，因为我们需要比较token值
            cudaStreamSynchronize(verify_stream);

            std::cout << "【验证】比较 - 目标模型: " << last_target_token_value << " vs 草稿模型[" << i
                      << "]: " << draft_token_value << std::endl;

            // 检查token是否匹配
            if (last_target_token_value != draft_token_value) {
                found_mismatch = i;

                std::cout << "【验证】不匹配! 位置: " << i << ", 目标模型: " << last_target_token_value
                          << ", 草稿模型: " << draft_token_value << std::endl;
                break;
            }

            // 如果token匹配，增加匹配长度
            max_match_length++;

            std::cout << "【验证】匹配成功! 位置: " << i << ", token: " << draft_token_value << std::endl;

            // 更新输入序列用于下一次验证
            input_sequence.push_back(last_target_token_value);
            verified_tokens.push_back(last_target_token_value);  // 直接添加到验证通过的token

            // 如果已经是最后一个token，不需要再生成下一个
            if (i == draft_tokens_gpu.size() - 2) {
                break;
            }

            // 准备验证下一个token
            // 直接使用GPU上的草稿token作为输入创建张量，避免额外的CPU-GPU拷贝
            Tensor<uint32_t> next_input(draft_tokens_gpu[i + 1], {1}, device_);

            // KV缓存已经提前扩展，只需要更新当前大小记录
            current_cache_size++;

            // 使用forward生成下一个token
            target_kv_cache_.resize(target_kv_cache_.size() + 1);
            uint32_t* next_token = target_model_->forward(&next_input, thread_pool_, &target_kv_cache_, top_k,
                                                          temperature, top_p, d_states);
            last_target_token = next_token;
            // 检查是否生成成功
            if (next_token == nullptr) {
                break;
            }

            // 获取目标模型生成的token值
            cudaMemcpyAsync(&last_target_token_value, next_token, sizeof(uint32_t), cudaMemcpyDeviceToHost,
                            verify_stream);
            cudaStreamSynchronize(verify_stream);  // 确保拷贝完成，这里必须同步
        }

        // 调整KV缓存大小为实际使用的大小
        // target_kv_cache_.resize(current_cache_size);
        std::cout << "是否匹配: " << found_mismatch << std::endl;
        // 如果有不匹配的token，将目标模型生成的替代token添加到列表中
        if (found_mismatch != -1 && last_target_token_value >= 0) {
            draft_kv_cache_.resize(draft_kv_cache_.size() - draft_tokens_gpu.size() + found_mismatch + 2);
            std::cout << "【验证】草稿模型回退KV缓存大小: " << draft_kv_cache_.size() << std::endl;
            verified_tokens.push_back(last_target_token_value);
        }

        return max_match_length;
    } catch (const std::exception& e) {
        std::cout << "GPU验证过程中出错: " << e.what() << std::endl;
        return 0;
    }
}

// 显式实例化模板类
template class SpeculativeDecoder<__nv_bfloat16>;
