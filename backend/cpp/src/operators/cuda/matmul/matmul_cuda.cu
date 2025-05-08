#include <cuda_runtime.h>

#include <stdexcept>

#include "cudaOP.cuh"
#include "operators/matmul/matmul_cuda.cuh"
#include "operators/matmul/matmul_selector.hpp"

namespace op {

// 实现MatmulCUDAOperator的operator()方法
template <typename T>
void MatmulCUDAOperator<T>::operator()(Tensor<T>* output, Tensor<T>* input, const WeightTensor<T>& weight,
                                       const Tensor<T>* bias, cudaStream_t stream) {
    // 调用包装函数，处理量化和非量化权重
    if (weight.is_quantized()) {
        // 使用AWQ量化矩阵乘法
        cuda_OP::matmul_quantized_gemv(*input, *weight.qweight(), *weight.scales(), *weight.qzeros(),
                                       weight.group_size(), output, stream, bias);
    } else {
        // 使用普通矩阵乘法
        cuda_OP::matmul(*input, *weight.tensor(), output, stream, bias);
    }
}

// 显式模板实例化
template class MatmulCUDAOperator<float>;
template class MatmulCUDAOperator<__nv_bfloat16>;

}  // namespace op