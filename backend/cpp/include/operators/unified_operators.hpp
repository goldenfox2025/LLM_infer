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
                fprintf(stderr, "CUDA initialization failed: %s. Falling back to CPU.\n", e.what());
                device_ = Device::CPU;
                platform_ = OperatorPlatform::CPU;
                OperatorFactory<T>::registerCPUOperators();
            }
#endif
        }
    }

    // 切换到CUDA设备
    void cuda() {
        if (device_ == Device::CUDA)
            return;

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
            fprintf(stderr, "CUDA initialization failed: %s. Falling back to CPU.\n", e.what());
            device_ = Device::CPU;
            platform_ = OperatorPlatform::CPU;
            OperatorFactory<T>::registerCPUOperators();
        }
#else
        fprintf(stderr, "Warning: CUDA support not compiled in. Using CPU instead.\n");
        device_ = Device::CPU;
        platform_ = OperatorPlatform::CPU;
#endif
    }

    // 切换到CPU设备
    void cpu() {
        if (device_ == Device::CPU)
            return;
        device_ = Device::CPU;
        platform_ = OperatorPlatform::CPU;
        OperatorFactory<T>::registerCPUOperators();
    }

    // RoPE算子
    void rope(Tensor<T>* tensor, size_t offset, float theta, cudaStream_t stream = nullptr) {
        auto op = OperatorFactory<T>::getRopeOperator(platform_);
        if (!op) {
            // 检查是否是CPU平台
            if (platform_ == OperatorPlatform::CPU) {
                // 检查是否是__nv_bfloat16类型
                if constexpr (std::is_same_v<T, __nv_bfloat16>) {
                    // 在实际应用中，这里应该有一个转换逻辑，将 __nv_bfloat16 转换为
                    // float， 然后调用 float 版本的算子，最后再转换回 __nv_bfloat16
                    // 但这里我们简单地抛出异常
                    throw std::runtime_error("RoPE operator for __nv_bfloat16 not supported on CPU platform");
                }
            }
            // 如果不是特殊情况，抛出通用错误
            throw std::runtime_error("RoPE operator not registered for the current platform");
        }

        // 直接调用算子
        (*op)(tensor, offset, theta, stream);
    }

    // RMS Norm算子
    void rms_norm(Tensor<T>* output, Tensor<T>* input, Tensor<T>* weight, float eps, cudaStream_t stream = nullptr) {
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
            throw std::runtime_error("RMS Norm operator not registered for the current platform");
        }

        // 直接调用算子
        (*op)(output, input, weight, eps, stream);
    }

    // Multiply算子
    void multiply(Tensor<T>* output, Tensor<T>* input_a, Tensor<T>* input_b, cudaStream_t stream = nullptr) {
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
            throw std::runtime_error("Multiply operator not registered for the current platform");
        }

        // 直接调用算子
        (*op)(output, input_a, input_b, stream);
    }

    // SiLU算子
    void silu(Tensor<T>* output, Tensor<T>* input, cudaStream_t stream = nullptr) {
        auto op = OperatorFactory<T>::getSiluOperator(platform_);
        if (!op) {
            // 检查是否是CPU平台
            if (platform_ == OperatorPlatform::CPU) {
                // 检查是否是__nv_bfloat16类型
                if constexpr (std::is_same_v<T, __nv_bfloat16>) {
                    throw std::runtime_error("SiLU operator for __nv_bfloat16 not supported on CPU platform");
                }
            }
            // 如果不是特殊情况，抛出通用错误
            throw std::runtime_error("SiLU operator not registered for the current platform");
        }

        // 直接调用算子
        (*op)(output, input, stream);
    }

    // Add算子
    void add(Tensor<T>* output, Tensor<T>* input_a, Tensor<T>* input_b, cudaStream_t stream = nullptr) {
        auto op = OperatorFactory<T>::getAddOperator(platform_);
        if (!op) {
            // 检查是否是CPU平台
            if (platform_ == OperatorPlatform::CPU) {
                // 检查是否是__nv_bfloat16类型
                if constexpr (std::is_same_v<T, __nv_bfloat16>) {
                    throw std::runtime_error("Add operator for __nv_bfloat16 not supported on CPU platform");
                }
            }
            // 如果不是特殊情况，抛出通用错误
            throw std::runtime_error("Add operator not registered for the current platform");
        }

        // 直接调用算子
        (*op)(output, input_a, input_b, stream);
    }
    void matmul(Tensor<T>* output, Tensor<T>* input, const WeightTensor<T>& weight, const Tensor<T>* bias = nullptr,
                cudaStream_t stream = nullptr) {
        // 根据权重类型自动选择合适的实现
        if (weight.is_quantized()) {
            // 使用AWQ量化矩阵乘法
            cuda_OP::matmul_quantized_gemv(*input, *weight.qweight(), *weight.scales(), *weight.qzeros(),
                                           weight.group_size(), output, stream, bias);
        } else {
            // 使用普通矩阵乘法
            cuda_OP::matmul(*input, *weight.tensor(), output, stream, bias);
        }
    }
    // 统一的矩阵乘法接口，支持普通权重和量化权重
    //     void matmul(Tensor<T>* output, Tensor<T>* input, const WeightTensor<T>& weight, const Tensor<T>* bias =
    //     nullptr,
    //                 cudaStream_t stream = nullptr) {
    //         auto op = OperatorFactory<T>::getMatmulOperator(platform_);
    //         if (!op) {
    //             // 检查是否是CPU平台
    //             if (platform_ == OperatorPlatform::CPU) {
    //                 // 检查是否是__nv_bfloat16类型
    //                 if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    //                     throw std::runtime_error("MatMul operator for __nv_bfloat16 not supported on CPU platform");
    //                 }
    //             }
    //             // 如果不是特殊情况，抛出通用错误
    //             throw std::runtime_error("MatMul operator not registered for the current platform");
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

    // 添加其他算子的接口...
};

}  // namespace op
