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

// 定义检查CUDA错误的宏
#define checkCudaErrors(call)                                           \
  do {                                                                  \
    cudaError_t err = call;                                             \
    if (err != cudaSuccess) {                                           \
      fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(err));                                 \
      throw std::runtime_error(cudaGetErrorString(err));                \
    }                                                                   \
  } while (0)

// 构造函数实现
template <typename T>
SpeculativeDecoder<T>::SpeculativeDecoder(
    std::shared_ptr<BaseModel> target_model,
    std::shared_ptr<BaseModel> draft_model, size_t spec_length)
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
      thread_pool_(4),
      device_(Device::CUDA),
      spec_length_(spec_length),
      d_states(nullptr) {
  // 确保两个模型都在CUDA上
  if (target_model_->device() != Device::CUDA) {
    target_model_->cuda();
  }
  if (draft_model_->device() != Device::CUDA) {
    draft_model_->cuda();
  }

  // 初始化CUDA资源
  init_cuda_resources();

  std::cout << "【投机解码器初始化完成】" << std::endl;
  std::cout << "目标模型: " << target_model_->get_n_layers() << " 层"
            << std::endl;
  std::cout << "草稿模型: " << draft_model_->get_n_layers() << " 层"
            << std::endl;
  std::cout << "投机长度: " << spec_length_ << std::endl;
}

// 初始化CUDA资源
template <typename T>
void SpeculativeDecoder<T>::init_cuda_resources() {
  // 分配CUDA随机状态
  cudaMalloc(&d_states, sizeof(curandState));
  int seed = std::chrono::system_clock::now().time_since_epoch().count();
  cuda_OP::init_curand(d_states, seed, 0, nullptr);
}

// 释放CUDA资源
template <typename T>
void SpeculativeDecoder<T>::free_cuda_resources() {
  if (d_states != nullptr) {
    cudaDeviceSynchronize();
    cudaFree(d_states);
    d_states = nullptr;
  }
}

// 使用草稿模型生成多个token
template <typename T>
GPUTokens SpeculativeDecoder<T>::generate_draft_tokens(
    const uint32_t* input_token, size_t num_tokens, float temperature,
    float top_p, size_t top_k) {
  std::cout << "开始使用草稿模型生成 " << num_tokens << " 个token" << std::endl;

  GPUTokens gpu_tokens;

  // 获取输入token值
  uint32_t input_token_value;
  cudaMemcpy(&input_token_value, input_token, sizeof(uint32_t),
             cudaMemcpyDeviceToHost);

  // 添加第一个token（输入token）
  gpu_tokens.add_token(const_cast<uint32_t*>(input_token), input_token_value);
  std::cout << "使用输入token: " << input_token_value << std::endl;

  // 限制生成token数量，避免卡死
  size_t max_tokens = std::min(num_tokens, (size_t)3);

  try {
    // 记录当前KV缓存大小
    size_t initial_kv_size = draft_kv_cache_.size();
    std::cout << "草稿模型初始KV缓存大小: " << initial_kv_size << std::endl;

    // 当前token指针，初始为输入token
    uint32_t* d_current_token = const_cast<uint32_t*>(input_token);

    // 生成token，使用forward（注意：KV缓存已经在之前通过prefill初始化）
    for (size_t i = 1; i < max_tokens; i++) {
      // 构造输入张量，使用当前token
      Tensor<uint32_t> input(d_current_token, {1}, device_);

      // 调整KV缓存大小
      draft_kv_cache_.resize(draft_kv_cache_.size() + 1);
      std::cout << "调整草稿模型KV缓存大小(forward): " << draft_kv_cache_.size()
                << std::endl;

      // 前向计算
      std::cout << "草稿模型生成第 " << i << " 个token..." << std::endl;
      uint32_t* next_token =
          draft_model_->forward(&input, thread_pool_, &draft_kv_cache_, top_k,
                                temperature, top_p, d_states);

      if (next_token == nullptr) {
        std::cout << "警告: 草稿模型forward返回了空指针，停止生成" << std::endl;
        break;
      }

      // 获取token值
      uint32_t token_value;
      cudaMemcpy(&token_value, next_token, sizeof(uint32_t),
                 cudaMemcpyDeviceToHost);
      std::cout << "草稿模型生成token: " << token_value << std::endl;

      // 添加生成的token
      gpu_tokens.add_token(next_token, token_value);

      // 更新当前token为新生成的token
      d_current_token = next_token;
    }

    // 验证KV缓存大小是否正确
    size_t expected_size =
        initial_kv_size + gpu_tokens.size() - 1;  // 减去输入token
    if (draft_kv_cache_.size() != expected_size) {
      std::cout << "警告: KV缓存大小异常，预期: " << expected_size
                << ", 实际: " << draft_kv_cache_.size() << std::endl;
      // 确保KV缓存大小正确
      draft_kv_cache_.resize(expected_size);
    }

  } catch (const std::exception& e) {
    std::cout << "草稿模型生成token时出错: " << e.what() << std::endl;
  }

  std::cout << "草稿模型共生成 " << gpu_tokens.size() << " 个token"
            << std::endl;
  return gpu_tokens;
}

// 验证草稿模型生成的token
template <typename T>
size_t SpeculativeDecoder<T>::verify_draft_tokens(
    const std::vector<uint32_t>& prefix_tokens,
    const std::vector<uint32_t>& draft_tokens, float temperature, float top_p,
    size_t top_k, std::vector<uint32_t>& verified_tokens) {
  // 记录最长匹配长度
  size_t max_match_length = 0;
  verified_tokens.clear();

  // 首先使用目标模型处理第一个token
  if (draft_tokens.empty()) {
    std::cout << "警告: 草稿模型未生成任何token" << std::endl;
    return 0;
  }

  // 如果只有一个token（输入token），直接返回
  if (draft_tokens.size() == 1) {
    std::cout << "草稿模型只生成了输入token，无需验证" << std::endl;
    return 0;
  }

  // 打印调试信息
  std::cout << "验证草稿模型生成的token: " << draft_tokens.size() << " 个"
            << std::endl;

  try {
    std::cout << "开始验证草稿模型生成的token..." << std::endl;

    // 获取草稿模型生成的token（不包括第一个，因为它是输入token）
    std::vector<uint32_t> draft_generated_tokens;
    for (size_t i = 1; i < draft_tokens.size(); i++) {
      draft_generated_tokens.push_back(draft_tokens[i]);
    }

    std::cout << "草稿模型生成token数量: " << draft_generated_tokens.size()
              << std::endl;
    std::cout << "草稿模型生成token: ";
    for (size_t i = 0; i < draft_generated_tokens.size() && i < 10; i++) {
      std::cout << draft_generated_tokens[i] << " ";
    }
    if (draft_generated_tokens.size() > 10) {
      std::cout << "... (截断显示)";
    }
    std::cout << std::endl;

    // 保存原始KV缓存大小和打印确认
    size_t kv_cache_initial_size = target_kv_cache_.size();
    std::cout << "目标模型初始KV缓存大小: " << kv_cache_initial_size
              << std::endl;

    // 创建用于验证的累积序列，初始为前缀序列
    std::vector<uint32_t> accumulated_sequence = prefix_tokens;
    std::cout << "初始累积序列长度: " << accumulated_sequence.size()
              << std::endl;

    // 记录最后一个目标模型生成的token
    uint32_t last_target_token_value = 0;
    bool found_mismatch = false;

    // 逐个验证草稿token
    for (size_t i = 0; i < draft_generated_tokens.size(); i++) {
      uint32_t draft_token_value = draft_generated_tokens[i];
      std::cout << "验证第 " << (i + 1) << " 个token: " << draft_token_value
                << std::endl;

      // 构建验证输入序列（取整个累积序列，为prefill做准备）
      std::vector<uint32_t> input_sequence = accumulated_sequence;

      // 打印当前输入序列的信息
      std::cout << "当前输入序列长度: " << input_sequence.size() << std::endl;
      if (!input_sequence.empty()) {
        std::cout << "输入序列最后一个token: " << input_sequence.back()
                  << std::endl;
      }

      // 将验证输入序列移动到GPU上（使用std::move避免拷贝）
      Tensor<uint32_t> input_tensor(std::move(input_sequence),
                                    {accumulated_sequence.size()}, device_);

      // 清除和调整目标模型的KV缓存大小为当前序列长度
      target_kv_cache_.clear();
      target_kv_cache_.resize(accumulated_sequence.size());
      std::cout << "重置并调整目标模型KV缓存大小: " << target_kv_cache_.size()
                << std::endl;

      // 使用prefill生成下一个token
      std::cout << "使用prefill验证token..." << std::endl;
      uint32_t* target_token =
          target_model_->prefill(&input_tensor, thread_pool_, &target_kv_cache_,
                                 top_k, temperature, top_p, d_states);

      // 检查是否生成成功
      if (target_token == nullptr) {
        std::cout << "错误: 目标模型生成token失败" << std::endl;
        break;
      }

      // 获取目标模型生成的token值
      uint32_t target_token_value;
      cudaMemcpy(&target_token_value, target_token, sizeof(uint32_t),
                 cudaMemcpyDeviceToHost);

      // 对目标模型生成的token进行有效性检查
      const uint32_t MAX_VALID_TOKEN_ID = 1900000;  // 一个合理的token ID上限
      if (target_token_value > MAX_VALID_TOKEN_ID) {
        std::cout << "警告: 目标模型生成的token值(" << target_token_value
                  << ")异常大，可能是指针地址而非有效token" << std::endl;

        // 分配GPU内存并使用备用forward方法尝试再次生成
        std::cout << "尝试使用forward方法重新生成..." << std::endl;
        if (!accumulated_sequence.empty()) {
          uint32_t last_token = accumulated_sequence.back();
          uint32_t* d_last_token;
          cudaMalloc(&d_last_token, sizeof(uint32_t));
          cudaMemcpy(d_last_token, &last_token, sizeof(uint32_t),
                     cudaMemcpyHostToDevice);

          // 创建输入张量
          Tensor<uint32_t> forward_input(d_last_token, {1}, device_);

          // 使用forward生成
          uint32_t* retry_token = target_model_->forward(
              &forward_input, thread_pool_, &target_kv_cache_, top_k,
              temperature, top_p, d_states);

          // 释放临时内存
          cudaFree(d_last_token);

          // 检查重试结果
          if (retry_token != nullptr) {
            cudaMemcpy(&target_token_value, retry_token, sizeof(uint32_t),
                       cudaMemcpyDeviceToHost);
            std::cout << "使用forward重新生成token: " << target_token_value
                      << std::endl;

            // 再次检查token有效性
            if (target_token_value > MAX_VALID_TOKEN_ID) {
              std::cout << "警告: 重新生成的token值仍然异常，跳过本轮验证"
                        << std::endl;
              break;
            }
          } else {
            std::cout << "使用forward重新生成失败，跳过本轮验证" << std::endl;
            break;
          }
        } else {
          std::cout << "累积序列为空，无法使用forward重新生成，跳过本轮验证"
                    << std::endl;
          break;
        }
      }

      std::cout << "目标模型生成token: " << target_token_value << std::endl;

      // 检查是否匹配
      std::cout << "检查token是否匹配: 目标=" << target_token_value
                << ", 草稿=" << draft_token_value << std::endl;

      if (target_token_value != draft_token_value) {
        std::cout << "Token不匹配，停止验证" << std::endl;
        // 保存最后一个目标模型生成的token（用于后续添加）
        last_target_token_value = target_token_value;
        found_mismatch = true;
        break;
      }

      // 更新最长匹配长度
      max_match_length = i + 1;

      // 将验证通过的token添加到累积序列中，为下一次验证做准备
      accumulated_sequence.push_back(target_token_value);

      // 保存最后一个目标模型生成的token
      last_target_token_value = target_token_value;
    }

    std::cout << "验证后KV缓存大小: " << target_kv_cache_.size() << std::endl;

    // 将验证通过的草稿模型token添加到验证通过的token列表中
    for (size_t i = 0;
         i < max_match_length && i < draft_generated_tokens.size(); i++) {
      verified_tokens.push_back(draft_generated_tokens[i]);
      std::cout << "添加验证通过的token " << i + 1 << ": "
                << draft_generated_tokens[i] << std::endl;
    }

    // 如果有不匹配的token，将目标模型生成的替代token添加到列表中
    // 确保验证失败时也会添加目标模型生成的token（关键改进）
    if (found_mismatch && last_target_token_value > 0) {
      std::cout << "添加目标模型生成的不匹配token: " << last_target_token_value
                << std::endl;
      verified_tokens.push_back(last_target_token_value);
    }

    std::cout << "验证完成，最长匹配长度: " << max_match_length << std::endl;
    std::cout << "验证通过的token数量: " << verified_tokens.size() << std::endl;

    return max_match_length;
  } catch (const std::exception& e) {
    std::cout << "验证过程中出错: " << e.what() << std::endl;
    return 0;
  }
}

// 生成文本的主要方法
template <typename T>
void SpeculativeDecoder<T>::generate_with_callback(
    const std::vector<uint32_t>& input_ids, size_t max_length,
    float temperature, float top_p, size_t top_k,
    std::function<void(uint32_t)> callback) {
  try {
    // 清空KV缓存
    target_kv_cache_.clear();
    draft_kv_cache_.clear();

    // 复制输入序列
    std::vector<uint32_t> current_ids = input_ids;

    // 初始预填充 - 目标模型
    std::cout << "执行目标模型初始预填充 (提示词长度: " << current_ids.size()
              << ")" << std::endl;

    // 确保输入序列非空
    if (current_ids.empty()) {
      std::cout << "错误: 输入序列为空" << std::endl;
      return;
    }

    // 将输入序列复制到GPU
    std::vector<uint32_t> input_vec = current_ids;  // 复制，确保不修改原始序列
    Tensor<uint32_t> input_tensor(std::move(input_vec), {current_ids.size()},
                                  device_);

    // 设置KV缓存大小和预填充
    target_kv_cache_.resize(target_kv_cache_.size() + current_ids.size());
    std::cout << "目标模型KV缓存大小(prefill前): " << target_kv_cache_.size()
              << std::endl;

    // 使用prefill处理整个提示词序列
    uint32_t* first_token =
        target_model_->prefill(&input_tensor, thread_pool_, &target_kv_cache_,
                               top_k, temperature, top_p, d_states);
    target_kv_cache_.resize(target_kv_cache_.size() + 1);
    std::cout << "目标模型KV缓存大小(prefill后): " << target_kv_cache_.size()
              << std::endl;

    if (first_token == nullptr) {
      std::cout << "警告: 目标模型初始预填充返回空指针" << std::endl;
      return;
    }

    // 回调第一个token
    uint32_t first_token_value;
    // 从GPU内存复制到CPU
    cudaMemcpy(&first_token_value, first_token, sizeof(uint32_t),
               cudaMemcpyDeviceToHost);

    callback(first_token_value);
    current_ids.push_back(first_token_value);
    std::cout << "目标模型生成第一个token: " << first_token_value << std::endl;

    // 初始预填充 - 草稿模型
    std::cout << "执行草稿模型初始预填充..." << std::endl;
    std::vector<uint32_t> draft_input_vec =
        current_ids;  // 复制，确保不修改原始序列
    Tensor<uint32_t> draft_input_tensor(std::move(draft_input_vec),
                                        {current_ids.size()}, device_);
    draft_kv_cache_.clear();
    draft_kv_cache_.resize(draft_kv_cache_.size() + current_ids.size());
    std::cout << "草稿模型KV缓存大小(prefill前): " << draft_kv_cache_.size()
              << std::endl;

    // 使用prefill处理整个提示词序列（包括目标模型生成的第一个token）
    uint32_t* draft_first_token = draft_model_->prefill(
        &draft_input_tensor, thread_pool_, &draft_kv_cache_, top_k, temperature,
        top_p, d_states);

    std::cout << "草稿模型KV缓存大小(prefill后): " << draft_kv_cache_.size()
              << std::endl;

    if (draft_first_token == nullptr) {
      std::cout << "警告: 草稿模型初始预填充返回空指针" << std::endl;
      return;
    }

    std::cout << "草稿模型初始预填充完成" << std::endl;

    // 主循环：投机解码
    std::cout << "\n=== 开始投机解码主循环 ===" << std::endl;

    // 限制最大生成长度，避免无限循环
    size_t max_iterations = 100;
    // std::min(max_length, (size_t)5);  // 减少最大迭代次数
    size_t iteration = 0;

    while (current_ids.size() < max_length && iteration < max_iterations) {
      try {
        std::cout << "\n--- 投机解码迭代 " << iteration << " ---" << std::endl;
        std::cout << "当前序列长度: " << current_ids.size() << std::endl;

        // 获取最后一个token
        uint32_t last_token = current_ids.back();
        std::cout << "最后一个token: " << last_token << std::endl;

        // 创建一个 GPU 上的 token 副本
        uint32_t* d_last_token;
        cudaMalloc(&d_last_token, sizeof(uint32_t));
        cudaMemcpy(d_last_token, &last_token, sizeof(uint32_t),
                   cudaMemcpyHostToDevice);

        // 使用草稿模型生成投机token
        GPUTokens gpu_draft_tokens = generate_draft_tokens(
            d_last_token, spec_length_, temperature, top_p, top_k);

        // 创建一个 CPU 上的 draft_tokens 副本，用于兼容现有代码
        std::vector<uint32_t> draft_tokens = gpu_draft_tokens.cpu_tokens;

        // 释放 GPU 内存
        cudaFree(d_last_token);

        // 打印草稿模型生成的token
        std::cout << "草稿模型生成token: ";
        for (size_t i = 0; i < draft_tokens.size(); i++) {
          std::cout << draft_tokens[i] << " ";
        }
        std::cout << std::endl;

        if (draft_tokens.size() <= 1) {
          std::cout << "草稿模型未生成有效token，使用标准解码" << std::endl;

          // 使用目标模型进行标准解码
          Tensor<uint32_t> input(&last_token, {1}, device_);

          // 在调用forward前，手动调整KV缓存大小
          target_kv_cache_.resize(target_kv_cache_.size() + 1);
          std::cout << "标准解码调整KV缓存大小: " << target_kv_cache_.size()
                    << std::endl;

          uint32_t* next_token =
              target_model_->forward(&input, thread_pool_, &target_kv_cache_,
                                     top_k, temperature, top_p, d_states);

          if (next_token != nullptr) {
            uint32_t token_value;
            cudaMemcpy(&token_value, next_token, sizeof(uint32_t),
                       cudaMemcpyDeviceToHost);
            current_ids.push_back(token_value);
            callback(token_value);
            std::cout << "标准解码生成token: " << token_value << std::endl;

            // 检查是否生成了EOS token
            if (token_value == target_model_->get_eos_token_id()) {
              std::cout << "生成了EOS token，结束生成" << std::endl;
              return;
            }
          } else {
            std::cout << "标准解码返回空指针，结束生成" << std::endl;
            return;
          }

          iteration++;
          continue;
        }

        // 验证草稿模型生成的token
        std::vector<uint32_t> verified_tokens;
        std::cout << "开始验证草稿模型生成的token..." << std::endl;
        size_t match_length =
            verify_draft_tokens(current_ids, draft_tokens, temperature, top_p,
                                top_k, verified_tokens);
        std::cout << "验证完成，最长匹配长度: " << match_length << std::endl;

        // 如果没有验证通过的token，使用标准解码
        if (verified_tokens.empty()) {
          std::cout << "没有验证通过的token，使用标准解码" << std::endl;

          // 使用目标模型进行标准解码
          Tensor<uint32_t> input(&last_token, {1}, device_);

          // 在调用forward前，手动调整KV缓存大小
          target_kv_cache_.resize(target_kv_cache_.size() + 1);
          std::cout << "标准解码调整KV缓存大小: " << target_kv_cache_.size()
                    << std::endl;

          uint32_t* next_token =
              target_model_->forward(&input, thread_pool_, &target_kv_cache_,
                                     top_k, temperature, top_p, d_states);

          if (next_token != nullptr) {
            uint32_t token_value;
            cudaMemcpy(&token_value, next_token, sizeof(uint32_t),
                       cudaMemcpyDeviceToHost);
            current_ids.push_back(token_value);
            callback(token_value);
            std::cout << "标准解码生成token: " << token_value << std::endl;

            // 检查是否生成了EOS token
            if (token_value == target_model_->get_eos_token_id()) {
              std::cout << "生成了EOS token，结束生成" << std::endl;
              return;
            }
          } else {
            std::cout << "标准解码返回空指针，结束生成" << std::endl;
            return;
          }

          iteration++;
          continue;
        }

        iteration++;

        // 打印KV缓存信息
        std::cout << "当前KV缓存大小: 目标模型=" << target_kv_cache_.size()
                  << ", 草稿模型=" << draft_kv_cache_.size() << std::endl;

        // 添加验证通过的token到当前序列
        std::cout << "添加验证通过的token到当前序列..." << std::endl;

        if (verified_tokens.empty()) {
          std::cout << "警告: 没有验证通过的token，跳过添加" << std::endl;
        } else {
          // 添加所有验证通过的token
          for (size_t i = 0; i < verified_tokens.size(); i++) {
            try {
              uint32_t token = verified_tokens[i];
              current_ids.push_back(token);
              callback(token);
              std::cout << "添加token: " << token << std::endl;

              // 检查是否生成了EOS token
              if (token == target_model_->get_eos_token_id()) {
                std::cout << "生成了EOS token，结束生成" << std::endl;
                return;
              }
            } catch (const std::exception& e) {
              std::cout << "添加token时出错: " << e.what() << std::endl;
              break;
            }
          }
        }

        std::cout << "当前序列长度: " << current_ids.size() << std::endl;
      } catch (const std::exception& e) {
        std::cout << "投机解码迭代中出错: " << e.what() << std::endl;

        // 尝试使用标准解码继续
        try {
          if (!current_ids.empty()) {
            uint32_t last_token = current_ids.back();
            Tensor<uint32_t> input(&last_token, {1}, device_);

            // 在调用forward前，手动调整KV缓存大小
            target_kv_cache_.resize(target_kv_cache_.size() + 1);
            std::cout << "错误恢复调整KV缓存大小: " << target_kv_cache_.size()
                      << std::endl;

            uint32_t* next_token =
                target_model_->forward(&input, thread_pool_, &target_kv_cache_,
                                       top_k, temperature, top_p, d_states);

            if (next_token != nullptr) {
              uint32_t token_value;
              cudaMemcpy(&token_value, next_token, sizeof(uint32_t),
                         cudaMemcpyDeviceToHost);
              current_ids.push_back(token_value);
              callback(token_value);
              std::cout << "错误恢复: 标准解码生成token: " << token_value
                        << std::endl;

              // 检查是否生成了EOS token
              if (token_value == target_model_->get_eos_token_id()) {
                std::cout << "生成了EOS token，结束生成" << std::endl;
                return;
              }
            } else {
              std::cout << "错误恢复失败，结束生成" << std::endl;
              return;
            }
          }
        } catch (const std::exception& e) {
          std::cout << "错误恢复过程中出错: " << e.what() << std::endl;
          return;
        }

        iteration++;
      }
    }
  } catch (const std::exception& e) {
    std::cout << "投机解码过程中出错: " << e.what() << std::endl;
  }
}

// 显式实例化模板类
template class SpeculativeDecoder<__nv_bfloat16>;
