#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <stdexcept>

#include "cudaOP.cuh"
#include "operators/matmul/cutlass_matmul_cuda.cuh"

namespace op {

// 实现CutlassMatmulCUDAOperator的operator()方法
template <typename T>
void CutlassMatmulCUDAOperator<T>::operator()(Tensor<T>* output, Tensor<T>* input, const WeightTensor<T>& weight,
                                              const Tensor<T>* bias, cudaStream_t stream) {
    // 确保权重不是量化的
    if (weight.is_quantized()) {
        throw std::runtime_error("CUTLASS MatMul does not support quantized weights");
    }

    // 执行矩阵乘法
    // 调用使用CUTLASS后端的矩阵乘法
    cuda_OP::matmul(*input, *weight.tensor(), output, stream, bias, 2);  // 2表示使用CUTLASS
}

// 显式模板实例化
template class CutlassMatmulCUDAOperator<float>;
template class CutlassMatmulCUDAOperator<__nv_bfloat16>;

}  // namespace op