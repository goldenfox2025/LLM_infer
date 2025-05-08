#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <stdexcept>

#include "cudaOP.cuh"
#include "operators/matmul/cublas_matmul_cuda.cuh"

namespace op {

// CublasMatmulCUDAOperator构造函数
template <typename T>
CublasMatmulCUDAOperator<T>::CublasMatmulCUDAOperator() : initialized_(false) {
    initialize();
}

// CublasMatmulCUDAOperator析构函数
template <typename T>
CublasMatmulCUDAOperator<T>::~CublasMatmulCUDAOperator() {
    destroy();
}

// 初始化cuBLAS句柄
template <typename T>
void CublasMatmulCUDAOperator<T>::initialize() {
    if (!initialized_) {
        cublasStatus_t status = cublasCreate(&handle_);
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to create cuBLAS handle");
        }
        initialized_ = true;
    }
}

// 销毁cuBLAS句柄
template <typename T>
void CublasMatmulCUDAOperator<T>::destroy() {
    if (initialized_) {
        cublasDestroy(handle_);
        initialized_ = false;
    }
}

// 实现CublasMatmulCUDAOperator的operator()方法
template <typename T>
void CublasMatmulCUDAOperator<T>::operator()(Tensor<T>* output, Tensor<T>* input, const WeightTensor<T>& weight,
                                             const Tensor<T>* bias, cudaStream_t stream) {
    // 确保权重不是量化的
    if (weight.is_quantized()) {
        throw std::runtime_error("cuBLAS MatMul does not support quantized weights");
    }

    if (!initialized_) {
        initialize();
    }

    // 设置流
    if (stream) {
        cublasSetStream(handle_, stream);
    }

    // 使用cuBLAS执行矩阵乘法
    cuda_OP::matmul(*input, *weight.tensor(), output, stream, bias, 1);  // 1表示使用cuBLAS
}

// 显式模板实例化
template class CublasMatmulCUDAOperator<float>;
template class CublasMatmulCUDAOperator<__nv_bfloat16>;

}  // namespace op