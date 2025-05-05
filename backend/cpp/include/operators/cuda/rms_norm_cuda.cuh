#pragma once

#include <cuda_runtime.h>

#include "operators/operator_base.hpp"

namespace op {

template <typename T>
class RmsNormCUDAOperator : public RmsNormOperator<T> {
 public:
  RmsNormCUDAOperator() = default;
  ~RmsNormCUDAOperator() override = default;

  // 实现CUDA版本的RMS Norm - 使用二重指针以支持CUDA图优化
  void operator()(Tensor<T>** output_ptr, Tensor<T>** input_ptr,
                  Tensor<T>** weight_ptr, float* eps_ptr,
                  cudaStream_t stream = nullptr) override;

  // 获取算子平台
  OperatorPlatform platform() const override { return OperatorPlatform::CUDA; }
};

}  // namespace op
