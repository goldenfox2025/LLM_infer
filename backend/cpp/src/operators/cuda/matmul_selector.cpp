#include "operators/cuda/matmul/matmul_selector.hpp"

#include "operators/cpu/matmul_cpu.hpp"
#include "operators/cuda/matmul/awq_matmul_cuda.cuh"
#include "operators/cuda/matmul/cublas_matmul_cuda.cuh"
#include "operators/cuda/matmul/cutlass_matmul_cuda.cuh"
#include "operators/cuda/matmul/matmul_cuda.cuh"

namespace op {

// 注册CUDA实现
template <typename T>
void MatmulSelector<T>::registerCudaImplementations() {
    // 默认实现
    registerImpl(MatmulType::DEFAULT, OperatorPlatform::CUDA, std::make_shared<MatmulCUDAOperator<T>>());

    // cuBLAS实现
    registerImpl(MatmulType::CUBLAS, OperatorPlatform::CUDA, std::make_shared<CublasMatmulCUDAOperator<T>>());

    // CUTLASS实现
    registerImpl(MatmulType::CUTLASS, OperatorPlatform::CUDA, std::make_shared<CutlassMatmulCUDAOperator<T>>());

    // AWQ实现
    registerImpl(MatmulType::AWQ, OperatorPlatform::CUDA, std::make_shared<AwqMatmulCUDAOperator<T>>());
}

// 注册CPU实现
template <typename T>
void MatmulSelector<T>::registerCpuImplementations() {
    // 默认CPU实现
    registerImpl(MatmulType::DEFAULT, OperatorPlatform::CPU, std::make_shared<MatmulCPUOperator<T>>());
}

// 显式模板实例化
template class MatmulSelector<float>;
template class MatmulSelector<__nv_bfloat16>;

}  // namespace op