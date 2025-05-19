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

// 构造函数实现
template <typename T>
SpeculativeDecoder<T>::SpeculativeDecoder(
    std::shared_ptr<BaseModel> target_model,
    std::shared_ptr<BaseModel> draft_model, size_t spec_length,
    size_t thread_count)
    : target_model_(target_model),
      draft_model_(draft_model),
      target_kv_cache_(
          target_model->get_n_layers(), target_model->get_max_seq_len(),
          target_model->get_head_dim() * target_model->get_n_kv_heads(),
          Device::CUDA),
      draft_kv_cache_(
          draft_model->get_n_layers(), draft_model->get_max_seq_len(),
          draft_model->get_head_dim() * draft_model->get_n_kv_heads(),
          Device::CUDA),
      thread_pool_(thread_count),  // 使用传入的线程数
      device_(Device::CUDA),
      spec_length_(6),
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

  // std::cout << "【投机解码器初始化完成】" << std::endl;
  // std::cout << "目标模型: " << target_model_->get_n_layers() << " 层" <<
  // std::endl; std::cout << "草稿模型: " << draft_model_->get_n_layers() << "
  // 层" << std::endl; std::cout << "投机长度: " << spec_length_ << std::endl;
  // std::cout << "线程池大小: " << thread_count << std::endl;
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
    d_reuse_token = static_cast<uint32_t*>(
        GlobalCudaMemoryPool::get_tagged_memory(kReuseTokenTag));
  } else {
    d_reuse_token =
        static_cast<uint32_t*>(GlobalCudaMemoryPool::allocate_tagged(
            kReuseTokenTag, sizeof(uint32_t)));
  }

  // 如果内存池分配失败，则使用普通的CUDA内存分配
  if (d_reuse_token == nullptr) {
    (cudaMalloc(&d_reuse_token, sizeof(uint32_t)));
  }

  // 分配草稿模型tokens的固定内存
  // 为spec_length_个token分配内存
  if (GlobalCudaMemoryPool::has_tag(kDraftTokensTag)) {
    d_draft_tokens = static_cast<uint32_t*>(
        GlobalCudaMemoryPool::get_tagged_memory(kDraftTokensTag));
  } else {
    d_draft_tokens =
        static_cast<uint32_t*>(GlobalCudaMemoryPool::allocate_tagged(
            kDraftTokensTag,
            sizeof(uint32_t) * (spec_length_ + 1)));  // +1 为输入token
  }

  // 如果内存池分配失败，则使用普通的CUDA内存分配
  if (d_draft_tokens == nullptr) {
    (cudaMalloc(&d_draft_tokens, sizeof(uint32_t) * (spec_length_ + 1)));
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
  if (d_reuse_token != nullptr &&
      !GlobalCudaMemoryPool::has_tag(kReuseTokenTag)) {
    cudaFree(d_reuse_token);
    d_reuse_token = nullptr;
  }

  // 释放草稿模型tokens的固定内存
  if (d_draft_tokens != nullptr &&
      !GlobalCudaMemoryPool::has_tag(kDraftTokensTag)) {
    cudaFree(d_draft_tokens);
    d_draft_tokens = nullptr;
  }
}

// 使用草稿模型生成多个token，直接返回GPU指针数组
template <typename T>
std::vector<uint32_t*> SpeculativeDecoder<T>::generate_draft_tokens_gpu(
    uint32_t* input_token, size_t num_tokens, float temperature, float top_p,
    size_t top_k) {
  // 预分配GPU指针数组，避免频繁扩容
  std::vector<uint32_t*> gpu_tokens;
  int initial_kv_size = draft_kv_cache_.size();

  try {
    // 将输入token复制到固定内存的第一个位置
    cudaMemcpyAsync(d_draft_tokens, input_token, sizeof(uint32_t),
                    cudaMemcpyDeviceToDevice, draft_stream_);

    // 添加输入token指针到结果数组
    gpu_tokens.push_back(d_draft_tokens);

    // 生成剩余token，使用forward进行自回归生成
    for (size_t i = 0; i < num_tokens; i++) {
      // 构造输入张量，使用当前token位置
      uint32_t* d_current_token = d_draft_tokens + i;
      Tensor<uint32_t> input(d_current_token, {1}, device_);

      // 扩展KV缓存
      draft_kv_cache_.resize(draft_kv_cache_.size() + 1);

      // 使用sample_to_fixed直接将结果写入固定内存的下一个位置
      uint32_t* next_token_ptr = d_draft_tokens + i + 1;

      // 获取logits并直接采样到固定内存位置
      auto qwen3 = std::dynamic_pointer_cast<Qwen3Model<T>>(draft_model_);
      if (qwen3) {
        // 如果是Qwen3模型，使用forward_cuda获取logits
        Tensor<T> logits = qwen3->forward_cuda(&input, &draft_kv_cache_);
        cuda_OP::sample_to_fixed(std::move(logits), next_token_ptr, temperature,
                                 top_p, top_k, d_states, draft_stream_);
      } else {
        // 否则使用普通forward
        uint32_t* next_token =
            draft_model_->forward(&input, thread_pool_, &draft_kv_cache_, top_k,
                                  temperature, top_p, d_states);
        if (next_token == nullptr) {
          // 如果生成失败，需要将KV缓存大小调整回实际使用的大小
          draft_kv_cache_.resize(initial_kv_size + i);
          break;
        }
        // 复制结果到固定内存
        cudaMemcpyAsync(next_token_ptr, next_token, sizeof(uint32_t),
                        cudaMemcpyDeviceToDevice, draft_stream_);
        // 释放原始token内存 - 使用异步释放避免同步点
        cudaFreeAsync(next_token, draft_stream_);
      }

      // 保存GPU指针，避免不必要的GPU-CPU数据传输
      gpu_tokens.push_back(next_token_ptr);

      // 只在最后一次迭代检查EOS token，避免频繁的同步
      if (i == num_tokens - 1) {
        // 检查是否需要提前停止生成（例如EOS token）
        uint32_t token_value;
        cudaMemcpyAsync(&token_value, next_token_ptr, sizeof(uint32_t),
                        cudaMemcpyDeviceToHost, draft_stream_);
        cudaStreamSynchronize(draft_stream_);

        // 如果是EOS token，提前结束生成
        if (token_value == draft_model_->get_eos_token_id()) {
          break;
        }
      }
    }
  } catch (const std::exception& e) {
    // 记录错误并返回已生成的token
    // std::cout << "草稿模型生成token时出错: " << e.what() << std::endl;
  }

  return gpu_tokens;
}

// 批处理批量从GPU读取token值
template <typename T>
void SpeculativeDecoder<T>::generate_with_callback(
    const std::vector<uint32_t>& input_ids, size_t max_length,
    float temperature, float top_p, size_t top_k,
    std::function<void(uint32_t)> callback) {
  try {
    std::vector<uint32_t> current_ids;
    current_ids.reserve(max_length);  // 提前分配最大可能的大小
    current_ids = input_ids;

    // 确保输入序列非空
    if (current_ids.empty()) {
      // std::cout << "错误: 输入序列为空" << std::endl;
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
      Tensor<uint32_t> input_tensor({current_ids.begin(), current_ids.end()},
                                    {current_ids.size()}, device_);
      uint32_t* first_token =
          target_model_->prefill(&input_tensor, thread_pool_, &target_kv_cache_,
                                 top_k, temperature, top_p, d_states);
      // // 再进行一次forward
      // target_kv_cache_.resize(target_kv_cache_.size() + 1);
      // uint32_t* first_token_forward = target_model_->forward(&input_tensor,
      // thread_pool_, &target_kv_cache_,
      //                                                        top_k,
      //                                                        temperature,
      //                                                        top_p,
      //                                                        d_states);
      if (first_token == nullptr) {
        // std::cout << "警告: 目标模型初始预填充返回空指针" << std::endl;
        return;
      }

      // 从GPU读取第一个token值 - 使用异步拷贝

      cudaMemcpyAsync(&first_target_token_value, first_token, sizeof(uint32_t),
                      cudaMemcpyDeviceToHost, main_stream_);
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
      // std::cout << "草稿模型初始化，KV缓存大小为: " << draft_kv_cache_.size()
      // << std::endl;
      // 使用prefill为草稿模型准备KV缓存 - 直接传递数据不创建副本
      Tensor<uint32_t> draft_input_tensor(
          {current_ids.begin(), current_ids.end()}, {current_ids.size()},
          device_);
      uint32_t* draft_first_token = draft_model_->prefill(
          &draft_input_tensor, thread_pool_, &draft_kv_cache_, top_k,
          temperature, top_p, d_states);

      // 从GPU读取第一个token值 - 使用异步拷贝
      cudaMemcpyAsync(&first_draft_token_value, draft_first_token,
                      sizeof(uint32_t), cudaMemcpyDeviceToHost, main_stream_);
      cudaStreamSynchronize(main_stream_);  // 确保拷贝完成
    }
    current_ids.push_back(first_target_token_value);
    // current_ids.pop_back();
    // std::cout << "草稿模型初始化完成，第一个token值为: "
    // << first_draft_token_value << std::endl;
    // 3. 不需要额外分配GPU内存，直接使用预分配的重用内存
    // 将最大迭代次数限制为剩余的生成长度
    size_t max_iterations = max_length - current_ids.size();
    size_t iteration = 0;

    // 主循环：投机解码
    while (current_ids.size() < max_length && iteration < max_iterations) {
      try {
        // 获得目标模型最后推理出的token
        const uint32_t last_token = current_ids.back();
        cudaMemcpyAsync(d_reuse_token, &last_token, sizeof(uint32_t),
                        cudaMemcpyHostToDevice, main_stream_);
        cudaStreamSynchronize(main_stream_);  // 确保拷贝完成
        std::vector<uint32_t*> gpu_tokens;
        gpu_tokens = generate_draft_tokens_gpu(d_reuse_token, spec_length_,
                                               temperature, top_p, top_k);

        // 可选：将草稿token组合成一个tensor用于调试或其他处理
        // Tensor<uint32_t> combined_tokens = combine_draft_tokens(gpu_tokens);

        // 验证草稿模型生成的token
        // 直接使用GPU指针进行验证，避免不必要的GPU-CPU数据传输
        std::vector<uint32_t> verified_tokens;
        verified_tokens.reserve(gpu_tokens.size());

        // 使用新的GPU版本的验证函数
        size_t match_length =
            verify_draft_tokens_gpu(current_ids, gpu_tokens, temperature, top_p,
                                    top_k, verified_tokens, main_stream_);

        // verified_tokens一定会包含至少一个token(要么是匹配的token，要么是目标模型生成的替代token)

        // 批量添加验证通过的token - 避免逐个添加的开销
        bool found_eos = false;

        // 检查是否有EOS token并找到位置
        size_t eos_pos = verified_tokens.size();
        // std::cout << "【验证】验证通过的token数量: " <<
        // verified_tokens.size()
        // << std::endl;
        for (size_t i = 0; i < verified_tokens.size(); i++) {
          if (verified_tokens[i] == target_model_->get_eos_token_id()) {
            eos_pos = i;
            found_eos = true;
            break;
          }
        }

        // 只添加到EOS标记处（如果有）
        size_t tokens_to_add = found_eos ? eos_pos + 1 : verified_tokens.size();
        current_ids.insert(current_ids.end(), verified_tokens.begin(),
                           verified_tokens.begin() + tokens_to_add);
        // std::cout << "【验证】当前输入序列: ";
        // for (size_t i = 0; i < current_ids.size(); i++) {
        //   std::cout << current_ids[i] << " ";
        // }
        // std::cout << std::endl;
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
        // std::cout << "投机解码迭代中出错: " << e.what() << std::endl;

        // 尝试使用标准解码继续
        try {
          if (!current_ids.empty()) {
            uint32_t last_token = current_ids.back();
            cudaMemcpyAsync(d_reuse_token, &last_token, sizeof(uint32_t),
                            cudaMemcpyHostToDevice, main_stream_);
            cudaStreamSynchronize(main_stream_);

            Tensor<uint32_t> input(d_reuse_token, {1}, device_);

            // 调整KV缓存大小
            target_kv_cache_.resize(target_kv_cache_.size() + 1);

            uint32_t* next_token =
                target_model_->forward(&input, thread_pool_, &target_kv_cache_,
                                       top_k, temperature, top_p, d_states);

            if (next_token != nullptr) {
              uint32_t token_value;
              cudaMemcpyAsync(&token_value, next_token, sizeof(uint32_t),
                              cudaMemcpyDeviceToHost, main_stream_);
              cudaStreamSynchronize(main_stream_);

              current_ids.push_back(token_value);
              callback(token_value);

              // 检查是否生成了EOS token
              if (token_value == target_model_->get_eos_token_id()) {
                break;
              }
            } else {
              // std::cout << "错误恢复失败，结束生成" << std::endl;
              break;
            }
          }
        } catch (const std::exception& e) {
          // std::cout << "错误恢复过程中出错: " << e.what() << std::endl;
          break;
        }

        iteration++;
      }
    }

  } catch (const std::exception& e) {
    // std::cout << "投机解码过程中出错: " << e.what() << std::endl;
  }
}

// 批量验证草稿模型生成的token - GPU指针版本
template <typename T>
size_t SpeculativeDecoder<T>::verify_draft_tokens_gpu(
    const std::vector<uint32_t>& prefix_tokens,
    std::vector<uint32_t*>& draft_tokens_gpu, float temperature, float top_p,
    size_t top_k, std::vector<uint32_t>& verified_tokens, cudaStream_t stream) {
  // 记录最长匹配长度
  size_t max_match_length = 0;
  verified_tokens.clear();

  // 检查输入有效性
  if (draft_tokens_gpu.empty() || draft_tokens_gpu.size() == 1) {
    return 0;  // 如果没有token或只有一个token（输入token），直接返回
  }

  auto qwen3 = std::dynamic_pointer_cast<Qwen3Model<T>>(target_model_);
  if (!qwen3) {
    throw std::runtime_error("目标模型必须是Qwen3Model类型");
  }

  try {
    // 使用验证专用流
    cudaStream_t verify_stream = verify_stream_;

    // 保存目标模型当前的KV缓存大小
    size_t original_cache_size = target_kv_cache_.size();

    // 移除最后一个token（这是下一个要生成的token，不是草稿token）
    draft_tokens_gpu.pop_back();

    // 获取草稿token数量
    size_t num_draft_tokens = draft_tokens_gpu.size() - 1;  // 减去输入token

    // 一次性验证所有草稿token
    // 将草稿token组合成一个tensor，形状为[seqlen]
    Tensor<uint32_t> combined_tokens = combine_draft_tokens(draft_tokens_gpu);

    // 扩展KV缓存以容纳新的token
    target_kv_cache_.resize(original_cache_size + combined_tokens.numel());

    // 使用内存池分配固定内存用于存储目标模型的采样结果
    static const std::string kTargetTokensTag = "spec_target_tokens";
    uint32_t* target_tokens = nullptr;

    // 检查是否已经有标记内存
    if (GlobalCudaMemoryPool::has_tag(kTargetTokensTag)) {
      target_tokens = static_cast<uint32_t*>(
          GlobalCudaMemoryPool::get_tagged_memory(kTargetTokensTag));
    }

    // 如果没有标记内存或者标记内存不活跃，重新分配
    if (target_tokens == nullptr) {
      target_tokens =
          static_cast<uint32_t*>(GlobalCudaMemoryPool::allocate_tagged(
              kTargetTokensTag, sizeof(uint32_t) * combined_tokens.numel()));
    }

    // 使用prefill_cuda获取logits并直接采样到固定内存
    Tensor<T> logits_tensor =
        qwen3->prefill_cuda(&combined_tokens, &target_kv_cache_);

    // 使用sample_batch_to_fixed直接将结果写入固定内存
    cuda_OP::sample_batch_to_fixed(std::move(logits_tensor), target_tokens,
                                   temperature, top_p, top_k, d_states,
                                   verify_stream);

    // 比较目标模型和草稿模型的token
    int found_mismatch = -1;
    uint32_t last_target_token_value = 0;

    // 分配主机内存用于批量复制token
    std::vector<uint32_t> host_target_tokens(num_draft_tokens);
    std::vector<uint32_t> host_draft_tokens(num_draft_tokens);

    // 批量复制目标模型生成的token到主机内存
    cudaMemcpyAsync(host_target_tokens.data(), target_tokens,
                    sizeof(uint32_t) * num_draft_tokens, cudaMemcpyDeviceToHost,
                    verify_stream);

    // 批量复制草稿模型生成的token到主机内存
    for (size_t i = 0; i < num_draft_tokens; i++) {
      cudaMemcpyAsync(&host_draft_tokens[i], draft_tokens_gpu[i + 1],
                      sizeof(uint32_t), cudaMemcpyDeviceToHost, verify_stream);
    }

    // 等待所有复制完成
    cudaStreamSynchronize(verify_stream);

    // 在主机内存中比较token
    for (size_t i = 0; i < num_draft_tokens; i++) {
      if (host_target_tokens[i] != host_draft_tokens[i]) {
        found_mismatch = i;
        last_target_token_value = host_target_tokens[i];
        break;
      }

      // 添加到验证通过的token列表
      verified_tokens.push_back(host_target_tokens[i]);
    }

    // 如果没有找到不匹配的token，设置最后一个token值
    if (found_mismatch == -1 && num_draft_tokens > 0) {
      last_target_token_value = host_target_tokens[num_draft_tokens - 1];
    }

    // 如果找到不匹配的token，调整KV缓存大小并添加目标模型生成的token
    if (found_mismatch != -1 && last_target_token_value >= 0) {
      draft_kv_cache_.resize(draft_kv_cache_.size() - num_draft_tokens +
                             found_mismatch + 1);
      target_kv_cache_.resize(original_cache_size + found_mismatch + 1);
      verified_tokens.push_back(last_target_token_value);
    }

    // 不需要释放标记内存，由内存池管理

    return max_match_length;
  } catch (const std::exception& e) {
    return 0;
  }
}
#include "common.hpp"
// 将草稿token GPU指针组合成一个tensor
template <typename T>
Tensor<uint32_t> SpeculativeDecoder<T>::combine_draft_tokens(
    std::vector<uint32_t*>& draft_tokens_gpu) {
  // 使用Tensor类的静态方法将GPU指针组合成一个tensor
  Tensor<uint32_t> combined_tokens =
      Tensor<uint32_t>::combine_gpu_ptrs(draft_tokens_gpu, device_);
  // std::cout << "【组合】已将 " << draft_tokens_gpu.size()
  // << " 个草稿token组合成tensor，形状为[" << combined_tokens.sizes()[0]
  // << "]" << std::endl;
  // debugPrintTensor(combined_tokens, "combined_tokens");
  return combined_tokens;
}

// 显式实例化模板类
template class SpeculativeDecoder<__nv_bfloat16>;
