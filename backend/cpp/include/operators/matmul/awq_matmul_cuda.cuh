#pragma once

#include <cuda_runtime.h>

#include "operators/matmul/matmul.hpp"

namespace op {

template <typename T>
class AwqMatmulCUDAOperator : public MatmulOperatorImpl<T> {
   public:
    AwqMatmulCUDAOperator() = default;
    ~AwqMatmulCUDAOperator() override = default;

    // 实现AWQ量化版本的MatMul - 使用一重指针
    void operator()(Tensor<T>* output, Tensor<T>* input, const WeightTensor<T>& weight, const Tensor<T>* bias = nullptr,
                    cudaStream_t stream = nullptr) override;

    // 获取算子平台
    OperatorPlatform platform() const override {
        return OperatorPlatform::CUDA;
    }

    // 获取MatMul算子实现类型
    MatmulType impl_type() const override {
        return MatmulType::AWQ;
    }
};

}  // namespace op