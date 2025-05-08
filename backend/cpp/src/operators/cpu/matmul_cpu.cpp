#include "operators/matmul/matmul_cpu.hpp"

#include <cuda_bf16.h>

#include <cmath>
#include <stdexcept>

namespace op {

// 实现MatmulCPUOperator的operator()方法
template <typename T>
void MatmulCPUOperator<T>::operator()(Tensor<T>* output, Tensor<T>* input, const WeightTensor<T>& weight,
                                      const Tensor<T>* bias, cudaStream_t stream) {
    // 确保权重不是量化的
    if (weight.is_quantized()) {
        throw std::runtime_error("CPU MatMul does not support quantized weights");
    }

    // 检查是否是BF16类型，CPU不支持BF16
    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        throw std::runtime_error("MatMul operator for __nv_bfloat16 not supported on CPU platform");
    } else {
        // 获取输入张量和权重张量的形状
        const auto& input_shape = input->sizes();
        const auto& weight_shape = weight.tensor()->sizes();

        // 检查维度是否匹配
        if (input_shape[1] != weight_shape[1]) {
            throw std::runtime_error("Input and weight dimensions mismatch for MatMul");
        }

        size_t M = input_shape[0];   // 输入批次大小
        size_t K = input_shape[1];   // 特征维度
        size_t N = weight_shape[0];  // 输出特征维度

        // 执行CPU矩阵乘法 (A[M,K] × B[N,K] = C[M,N])
        for (size_t m = 0; m < M; ++m) {
            for (size_t n = 0; n < N; ++n) {
                float sum = 0.0f;

                // 计算点积
                for (size_t k = 0; k < K; ++k) {
                    sum += static_cast<float>(input->data_ptr()[m * K + k]) *
                           static_cast<float>(weight.tensor()->data_ptr()[n * K + k]);
                }

                // 添加偏置
                if (bias) {
                    sum += static_cast<float>(bias->data_ptr()[n]);
                }

                // 存储结果
                output->data_ptr()[m * N + n] = static_cast<T>(sum);
            }
        }
    }
}

// 显式模板实例化
template class MatmulCPUOperator<float>;
template class MatmulCPUOperator<__nv_bfloat16>;

}  // namespace op