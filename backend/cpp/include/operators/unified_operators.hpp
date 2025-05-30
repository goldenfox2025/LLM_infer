#pragma once

#include <cuda_bf16.h>

#include <memory>
#include <type_traits>

#include "operators/operator_base.hpp"
#include "operators/operator_factory.hpp"
#include "tensor.hpp"
#include "weight_tensor.hpp"

// 条件包含CUDA资源管理器
#if defined(__CUDACC__) || defined(__CUDA_ARCH__) || defined(__CUDA__)
#include "operators/cuda/cuda_resource_manager.cuh"
#endif

namespace op {

// 统一算子接口
template <typename T>
class UnifiedOperators {
 public:
  UnifiedOperators(Device device = Device::CPU) : device_(device) {
    // 根据设备类型初始化对应的算子
    if (device_ == Device::CPU) {
      OperatorFactory<T>::registerCPUOperators();
      platform_ = OperatorPlatform::CPU;
    } else {
      OperatorFactory<T>::registerCUDAOperators();
      platform_ = OperatorPlatform::CUDA;

      // 在CUDA设备上初始化CUDA资源
#if defined(__CUDACC__) || defined(__CUDA_ARCH__) || defined(__CUDA__)
      try {
        // 确保CUDAResourceManager已初始化
        auto& resource_manager = CUDAResourceManager::instance();
        // 预热，确保句柄创建
        resource_manager.getCublasHandle();
      } catch (const std::exception& e) {
        // 如果CUDA初始化失败，记录错误并回退到CPU
        fprintf(stderr,
                "CUDA initialization failed: %s. Falling back to CPU.\n",
                e.what());
        device_ = Device::CPU;
        platform_ = OperatorPlatform::CPU;
        OperatorFactory<T>::registerCPUOperators();
      }
#endif
    }
  }

  // 切换到CUDA设备
  void cuda() {
    if (device_ == Device::CUDA) return;

    device_ = Device::CUDA;
    platform_ = OperatorPlatform::CUDA;
    OperatorFactory<T>::registerCUDAOperators();

    // 初始化CUDA资源
#if defined(__CUDACC__) || defined(__CUDA_ARCH__) || defined(__CUDA__)
    try {
      // 确保CUDAResourceManager已初始化
      auto& resource_manager = CUDAResourceManager::instance();
      // 预热，确保句柄创建
      resource_manager.getCublasHandle();
    } catch (const std::exception& e) {
      // 如果CUDA初始化失败，记录错误并回退到CPU
      fprintf(stderr, "CUDA initialization failed: %s. Falling back to CPU.\n",
              e.what());
      device_ = Device::CPU;
      platform_ = OperatorPlatform::CPU;
      OperatorFactory<T>::registerCPUOperators();
    }
#else
    fprintf(stderr,
            "Warning: CUDA support not compiled in. Using CPU instead.\n");
    device_ = Device::CPU;
    platform_ = OperatorPlatform::CPU;
#endif
  }

  // 切换到CPU设备
  void cpu() {
    if (device_ == Device::CPU) return;
    device_ = Device::CPU;
    platform_ = OperatorPlatform::CPU;
    OperatorFactory<T>::registerCPUOperators();
  }

  // RoPE算子
  void rope(Tensor<T>* tensor, size_t offset, float theta,
            cudaStream_t stream = nullptr) {
    auto op = OperatorFactory<T>::getRopeOperator(platform_);
    if (!op) {
      // 检查是否是CPU平台
      if (platform_ == OperatorPlatform::CPU) {
        // 检查是否是__nv_bfloat16类型
        if constexpr (std::is_same_v<T, __nv_bfloat16>) {
          // 在实际应用中，这里应该有一个转换逻辑，将 __nv_bfloat16 转换为
          // float， 然后调用 float 版本的算子，最后再转换回 __nv_bfloat16
          // 但这里我们简单地抛出异常
          throw std::runtime_error(
              "RoPE operator for __nv_bfloat16 not supported on CPU platform");
        }
      }
      // 如果不是特殊情况，抛出通用错误
      throw std::runtime_error(
          "RoPE operator not registered for the current platform");
    }

    // 直接调用算子
    (*op)(tensor, offset, theta, stream);
  }

  // RMS Norm算子
  void rms_norm(Tensor<T>* output, Tensor<T>* input, Tensor<T>* weight,
                float eps, cudaStream_t stream = nullptr) {
    auto op = OperatorFactory<T>::getRmsNormOperator(platform_);
    if (!op) {
      // 检查是否是CPU平台
      if (platform_ == OperatorPlatform::CPU) {
        // 检查是否是__nv_bfloat16类型
        if constexpr (std::is_same_v<T, __nv_bfloat16>) {
          throw std::runtime_error(
              "RMS Norm operator for __nv_bfloat16 not supported on CPU "
              "platform");
        }
      }
      // 如果不是特殊情况，抛出通用错误
      throw std::runtime_error(
          "RMS Norm operator not registered for the current platform");
    }

    // 直接调用算子
    (*op)(output, input, weight, eps, stream);
  }

  // Multiply算子
  void multiply(Tensor<T>* output, Tensor<T>* input_a, Tensor<T>* input_b,
                cudaStream_t stream = nullptr) {
    auto op = OperatorFactory<T>::getMultiplyOperator(platform_);
    if (!op) {
      // 检查是否是CPU平台
      if (platform_ == OperatorPlatform::CPU) {
        // 检查是否是__nv_bfloat16类型
        if constexpr (std::is_same_v<T, __nv_bfloat16>) {
          throw std::runtime_error(
              "Multiply operator for __nv_bfloat16 not supported on CPU "
              "platform");
        }
      }
      // 如果不是特殊情况，抛出通用错误
      throw std::runtime_error(
          "Multiply operator not registered for the current platform");
    }

    // 直接调用算子
    (*op)(output, input_a, input_b, stream);
  }

  // SiLU算子
  void silu(Tensor<T>* output, Tensor<T>* input,
            cudaStream_t stream = nullptr) {
    auto op = OperatorFactory<T>::getSiluOperator(platform_);
    if (!op) {
      // 检查是否是CPU平台
      if (platform_ == OperatorPlatform::CPU) {
        // 检查是否是__nv_bfloat16类型
        if constexpr (std::is_same_v<T, __nv_bfloat16>) {
          throw std::runtime_error(
              "SiLU operator for __nv_bfloat16 not supported on CPU platform");
        }
      }
      // 如果不是特殊情况，抛出通用错误
      throw std::runtime_error(
          "SiLU operator not registered for the current platform");
    }

    // 直接调用算子
    (*op)(output, input, stream);
  }

  // Add算子
  void add(Tensor<T>* output, Tensor<T>* input_a, Tensor<T>* input_b,
           cudaStream_t stream = nullptr) {
    auto op = OperatorFactory<T>::getAddOperator(platform_);
    if (!op) {
      // 检查是否是CPU平台
      if (platform_ == OperatorPlatform::CPU) {
        // 检查是否是__nv_bfloat16类型
        if constexpr (std::is_same_v<T, __nv_bfloat16>) {
          throw std::runtime_error(
              "Add operator for __nv_bfloat16 not supported on CPU platform");
        }
      }
      // 如果不是特殊情况，抛出通用错误
      throw std::runtime_error(
          "Add operator not registered for the current platform");
    }

    // 直接调用算子
    (*op)(output, input_a, input_b, stream);
  }
  void matmul(Tensor<T>* output, Tensor<T>* input,
              const WeightTensor<T>& weight, const Tensor<T>* bias = nullptr,
              cudaStream_t stream = nullptr) {
    // 根据权重类型自动选择合适的实现
    if (weight.is_quantized()) {
      // 使用AWQ量化矩阵乘法
      cuda_OP::matmul_quantized_gemv(*input, *weight.qweight(),
                                     *weight.scales(), *weight.qzeros(),
                                     weight.group_size(), output, stream, bias);
    } else {
      // 使用普通矩阵乘法
      cuda_OP::matmul(*input, *weight.tensor(), output, stream, bias);
    }
  }
  // 统一的矩阵乘法接口，支持普通权重和量化权重
  //     void matmul(Tensor<T>* output, Tensor<T>* input, const WeightTensor<T>&
  //     weight, const Tensor<T>* bias = nullptr,
  //                 cudaStream_t stream = nullptr) {
  //         auto op = OperatorFactory<T>::getMatmulOperator(platform_);
  //         if (!op) {
  //             // 检查是否是CPU平台
  //             if (platform_ == OperatorPlatform::CPU) {
  //                 // 检查是否是__nv_bfloat16类型
  //                 if constexpr (std::is_same_v<T, __nv_bfloat16>) {
  //                     throw std::runtime_error("MatMul operator for
  //                     __nv_bfloat16 not supported on CPU platform");
  //                 }
  //             }
  //             // 如果不是特殊情况，抛出通用错误
  //             throw std::runtime_error("MatMul operator not registered for
  //             the current platform");
  //         }

  //         // 如果在CUDA平台上，确保设置了正确的流
  // #if defined(__CUDACC__) || defined(__CUDA_ARCH__) || defined(__CUDA__)
  //         if (platform_ == OperatorPlatform::CUDA && stream) {
  //             CUDAResourceManager::instance().setCublasStream(stream);
  //         }
  // #endif

  //         // 直接调用算子
  //         (*op)(output, input, weight, bias, stream);
  //     }

 private:
  Device device_;
  OperatorPlatform platform_;

 public:
  // 从嵌入表中根据索引获取嵌入向量
  void gather(Tensor<T>* output, const Tensor<uint32_t>* input,
              const Tensor<T>* embedding_table, cudaStream_t stream = nullptr) {
    // 目前直接调用 cuda_OP::gather
    if (device_ != Device::CUDA) {
      throw std::runtime_error(
          "Gather operator currently only supported on CUDA device");
    }
    cuda_OP::gather(output, input, embedding_table, stream);
  }

  // 从logits中采样下一个token
  uint32_t* sample(Tensor<T>&& logits, float temperature, float top_p,
                   size_t top_k, curandState* d_states,
                   cudaStream_t stream = nullptr) {
    // 目前直接调用 cuda_OP::sample
    if (device_ != Device::CUDA) {
      throw std::runtime_error(
          "Sample operator currently only supported on CUDA device");
    }
    return cuda_OP::sample(std::move(logits), temperature, top_p, top_k,
                           d_states, stream);
  }

  // 动态Flash Attention包装函数
  void dynamic_flash_attention(Tensor<T>& Q, const Tensor<T>& K,
                               const Tensor<T>& V, Tensor<T>& output,
                               int n_kv_heads, cudaStream_t stream = nullptr) {
    // 目前直接调用 cuda_OP::dynamic_flash_attention_wrapper
    if (device_ != Device::CUDA) {
      throw std::runtime_error(
          "Dynamic Flash Attention operator currently only supported on CUDA "
          "device");
    }
    cuda_OP::dynamic_flash_attention_wrapper(Q, K, V, output, n_kv_heads,
                                             stream);
  }

  // Prefill阶段计算注意力分数
  void compute_attention_scores_prefill(const Tensor<T>& Q, const Tensor<T>& K,
                                        Tensor<T>& att_scores, size_t head_dim,
                                        cudaStream_t stream = nullptr) {
    // 目前直接调用 cuda_OP::compute_attention_scores_prefill
    if (device_ != Device::CUDA) {
      throw std::runtime_error(
          "Compute Attention Scores Prefill operator currently only supported "
          "on CUDA device");
    }
    cuda_OP::compute_attention_scores_prefill(Q, K, att_scores, head_dim,
                                              stream);
  }

  // Prefill阶段计算注意力输出
  void compute_attention_output_prefill(const Tensor<T>& att_scores,
                                        const Tensor<T>& V,
                                        Tensor<T>& att_output, size_t n_heads,
                                        size_t head_dim, size_t total_seq_len,
                                        size_t n_kv_heads,
                                        cudaStream_t stream = nullptr) {
    // 目前直接调用 cuda_OP::compute_att_output_prefill
    if (device_ != Device::CUDA) {
      throw std::runtime_error(
          "Compute Attention Output Prefill operator currently only supported "
          "on CUDA device");
    }
    cuda_OP::compute_att_output_prefill(att_scores, V, att_output, n_heads,
                                        head_dim, total_seq_len, n_kv_heads,
                                        stream);
  }

  // Softmax算子
  void softmax(Tensor<T>* output, const Tensor<T>* input, int dim,
               bool mask = false, int offset = 0,
               cudaStream_t stream = nullptr) {
    // 目前直接调用 cuda_OP::softmax
    if (device_ != Device::CUDA) {
      throw std::runtime_error(
          "Softmax operator currently only supported on CUDA device");
    }
    cuda_OP::softmax(output, input, dim, mask, offset, stream);
  }

  // Flash Attention Prefill算子
  void flash_attention_prefill(const Tensor<T>& Q, const Tensor<T>& K,
                               const Tensor<T>& V, Tensor<T>& output,
                               int n_heads, int n_kv_heads, int head_dim,
                               int seq_len, int total_seq_len, int offset,
                               cudaStream_t stream = nullptr) {
    // 目前直接调用 cuda_OP::flash_attention_prefill
    if (device_ != Device::CUDA) {
      throw std::runtime_error(
          "Flash Attention Prefill operator currently only supported on CUDA "
          "device");
    }
    cuda_OP::flash_attention_prefill(Q, K, V, output, n_heads, n_kv_heads,
                                     head_dim, seq_len, total_seq_len, offset,
                                     stream);
  }
};

}  // namespace op
