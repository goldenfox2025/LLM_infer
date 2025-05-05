#pragma once

#include "operators/operator_base.hpp"

namespace op {

template <typename T>
class MultiplyCPUOperator : public MultiplyOperator<T> {
 public:
  MultiplyCPUOperator() = default;
  ~MultiplyCPUOperator() override = default;

  // 实现CPU版本的Multiply - 使用二重指针以支持CUDA图优化
  void operator()(Tensor<T>** output_ptr, Tensor<T>** input_a_ptr,
                  Tensor<T>** input_b_ptr,
                  cudaStream_t stream = nullptr) override {
    // 检查是否是BF16类型，CPU不支持BF16
    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      throw std::runtime_error(
          "Multiply operator for __nv_bfloat16 not supported on CPU platform");
    } else {
      // 从二重指针获取实际值
      Tensor<T>* output = *output_ptr;
      Tensor<T>* input_a = *input_a_ptr;
      Tensor<T>* input_b = *input_b_ptr;

      // 获取输入张量的大小
      size_t total = input_a->numel();

      // 检查输入张量的大小是否一致
      if (input_b->numel() != total) {
        throw std::runtime_error(
            "Multiply operator: input tensors must have the same size");
      }

      // 逐元素相乘
      for (size_t i = 0; i < total; i++) {
        output->data_ptr()[i] = input_a->data_ptr()[i] * input_b->data_ptr()[i];
      }
    }
  }

  // 获取算子平台
  OperatorPlatform platform() const override { return OperatorPlatform::CPU; }
};

}  // namespace op
