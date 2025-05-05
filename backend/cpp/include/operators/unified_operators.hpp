#pragma once

#include <cuda_bf16.h>

#include <memory>
#include <type_traits>

#include "operators/operator_base.hpp"
#include "operators/operator_factory.hpp"
#include "tensor.hpp"

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
    if (device_ == Device::CUDA) return;
    device_ = Device::CUDA;
    platform_ = OperatorPlatform::CUDA;
    OperatorFactory<T>::registerCUDAOperators();
  }

  // 切换到CPU设备
  void cpu() {
    if (device_ == Device::CPU) return;
    device_ = Device::CPU;
    platform_ = OperatorPlatform::CPU;
    OperatorFactory<T>::registerCPUOperators();
  }

  // RoPE算子 - 使用二重指针以支持CUDA图优化
  void rope(Tensor<T>* tensor, size_t offset, float theta,
            cudaStream_t stream = nullptr) {
    // 存储指针，以便通过二重指针传递
    tensor_ptr_ = tensor;
    offset_ptr_ = offset;

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

    // 通过二重指针调用算子
    (*op)(&tensor_ptr_, &offset_ptr_, theta, stream);
  }

  // RMS Norm算子 - 使用二重指针以支持CUDA图优化
  void rms_norm(Tensor<T>* output, Tensor<T>* input, Tensor<T>* weight,
                float eps, cudaStream_t stream = nullptr) {
    // 存储指针，以便通过二重指针传递
    output_ptr_ = output;
    input_ptr_ = input;
    weight_ptr_ = weight;
    eps_ptr_ = eps;

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

    // 通过二重指针调用算子
    (*op)(&output_ptr_, &input_ptr_, &weight_ptr_, &eps_ptr_, stream);
  }

 private:
  Device device_;
  OperatorPlatform platform_;

  // 用于存储指针的成员变量，以支持二重指针传递
  // RoPE算子
  Tensor<T>* tensor_ptr_;
  size_t offset_ptr_;

  // RMS Norm算子
  Tensor<T>* output_ptr_;
  Tensor<T>* input_ptr_;
  Tensor<T>* weight_ptr_;
  float eps_ptr_;

  // 添加其他算子的接口...
};

}  // namespace op
