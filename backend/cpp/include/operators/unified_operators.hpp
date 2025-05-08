#pragma once

#include <cuda_bf16.h>

#include <memory>
#include <type_traits>

#include "operators/operator_base.hpp"
#include "operators/operator_factory.hpp"
#include "tensor.hpp"
#include "weight_tensor.hpp"

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
        }
    }

    // 切换到CUDA设备
    void cuda() {
        if (device_ == Device::CUDA)
            return;
        device_ = Device::CUDA;
        platform_ = OperatorPlatform::CUDA;
        OperatorFactory<T>::registerCUDAOperators();
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

    // 统一的矩阵乘法接口，支持普通权重和量化权重
    void matmul(Tensor<T>* output, Tensor<T>* input, const WeightTensor<T>& weight, const Tensor<T>* bias = nullptr,
                cudaStream_t stream = nullptr) {
        auto op = OperatorFactory<T>::getMatmulOperator(platform_);
        if (!op) {
            // 检查是否是CPU平台
            if (platform_ == OperatorPlatform::CPU) {
                // 检查是否是__nv_bfloat16类型
                if constexpr (std::is_same_v<T, __nv_bfloat16>) {
                    throw std::runtime_error("MatMul operator for __nv_bfloat16 not supported on CPU platform");
                }
            }
            // 如果不是特殊情况，抛出通用错误
            throw std::runtime_error("MatMul operator not registered for the current platform");
        }

        // 直接调用算子
        (*op)(output, input, weight, bias, stream);
    }

   private:
    Device device_;
    OperatorPlatform platform_;

    // 添加其他算子的接口...
};

}  // namespace op
