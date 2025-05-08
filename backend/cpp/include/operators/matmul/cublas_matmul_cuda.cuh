#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "operators/matmul/matmul.hpp"

namespace op {

template <typename T>
class CublasMatmulCUDAOperator : public MatmulOperatorImpl<T> {
   public:
    CublasMatmulCUDAOperator();
    ~CublasMatmulCUDAOperator() override;

    // 实现cuBLAS版本的MatMul - 使用一重指针
    void operator()(Tensor<T>* output, Tensor<T>* input, const WeightTensor<T>& weight, const Tensor<T>* bias = nullptr,
                    cudaStream_t stream = nullptr) override;

    // 获取算子平台
    OperatorPlatform platform() const override {
        return OperatorPlatform::CUDA;
    }

    // 获取MatMul算子实现类型
    MatmulType impl_type() const override {
        return MatmulType::CUBLAS;
    }

   private:
    cublasHandle_t handle_;
    bool initialized_;

    // 初始化cuBLAS句柄
    void initialize();
    // 销毁cuBLAS句柄
    void destroy();
};

}  // namespace op