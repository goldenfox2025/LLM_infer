#include <stdexcept>

#include "advanced_kernels.cuh"
#include "cudaOP.cuh"

namespace cuda_OP {

// 预设的模板参数配置（减少共享内存使用）
constexpr int BM = 64;  // 从128减到64
constexpr int BN = 64;  // 从128减到64
constexpr int BK = 16;  // 从64减到32
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;  // kernel6使用16x16x16
constexpr int WMMA_K = 16;

constexpr int K_STAGE = 2;
constexpr int WARP_TILE_M = 1;
constexpr int WARP_TILE_N = 1;
constexpr int WAPR_NUM = BM * BN / 16 / 16 / WARP_TILE_M / WARP_TILE_N;
// kernel5: 16x8x16 MMA指令，不带bias
template <typename T>
void advanced_matmul_kernel5(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> *C, cudaStream_t stream) {
    const std::vector<size_t> &A_shape = A.sizes();
    const std::vector<size_t> &B_shape = B.sizes();

    size_t M = A_shape[0];
    size_t K = A_shape[1];
    size_t N = B_shape[1];

    // 确保矩阵维度合适
    if (A_shape[1] != B_shape[1]) {
        throw std::runtime_error("Matrix dimensions do not match for kernel5");
    }

    // 计算grid和block尺寸
    dim3 threadsPerBlock(128);  // 4个warp
    dim3 numBlocks((M + BM - 1) / BM, (N + BN - 1) / BN);

    // 启动kernel5，使用16x8x16 MMA
    constexpr int WMMA_N_K5 = 8;  // kernel5使用16x8x16
    advanced_kernels::kernel5<T, BM, BN, BK, WMMA_M, WMMA_N_K5, WMMA_K, WAPR_NUM, K_STAGE, WARP_TILE_M, WARP_TILE_N>
        <<<numBlocks, threadsPerBlock, 0, stream>>>(A.data_ptr(), B.data_ptr(), C->data_ptr(), M, N, K);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel5 launch failed: " + std::string(cudaGetErrorString(err)));
    }
}

// kernel5: 16x8x16 MMA指令，带bias
template <typename T>
void advanced_matmul_kernel5_with_bias(const Tensor<T> &A, const Tensor<T> &B, const Tensor<T> &bias, Tensor<T> *C,
                                       cudaStream_t stream) {
    const std::vector<size_t> &A_shape = A.sizes();
    const std::vector<size_t> &B_shape = B.sizes();
    const std::vector<size_t> &bias_shape = bias.sizes();

    size_t M = A_shape[0];
    size_t K = A_shape[1];
    size_t N = B_shape[1];

    // 检查维度匹配
    if (A_shape[1] != B_shape[1]) {
        throw std::runtime_error("Matrix dimensions do not match for kernel5");
    }
    if (bias_shape.size() != 1 || bias_shape[0] != N) {
        throw std::runtime_error("Bias dimensions do not match for kernel5");
    }

    // 计算grid和block尺寸
    dim3 threadsPerBlock(128);
    dim3 numBlocks((M + BM - 1) / BM, (N + BN - 1) / BN);

    // 启动kernel5 with bias
    constexpr int WMMA_N_K5 = 8;  // kernel5使用16x8x16
    advanced_kernels::kernel5_with_bias<T, BM, BN, BK, WMMA_M, WMMA_N_K5, WMMA_K, WAPR_NUM, K_STAGE, WARP_TILE_M,
                                        WARP_TILE_N><<<numBlocks, threadsPerBlock, 0, stream>>>(
        A.data_ptr(), B.data_ptr(), bias.data_ptr(), C->data_ptr(), M, N, K);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel5 with bias launch failed: " + std::string(cudaGetErrorString(err)));
    }
}

// kernel6: 16x16x16 MMA指令，不带bias
template <typename T>
void advanced_matmul_kernel6(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> *C, cudaStream_t stream) {
    const std::vector<size_t> &A_shape = A.sizes();
    const std::vector<size_t> &B_shape = B.sizes();

    size_t M = A_shape[0];
    size_t K = A_shape[1];
    size_t N = B_shape[1];

    // 计算grid和block尺寸
    dim3 threadsPerBlock(WAPR_NUM * 32);
    dim3 numBlocks((M + BM - 1) / BM, (N + BN - 1) / BN);

    // 启动kernel6，使用16x16x16 MMA
    advanced_kernels::kernel6<T, BM, BN, BK, WMMA_M, WMMA_N, WMMA_K, WAPR_NUM, K_STAGE, WARP_TILE_M, WARP_TILE_N>
        <<<numBlocks, threadsPerBlock, 0, stream>>>(A.data_ptr(), B.data_ptr(), C->data_ptr(), M, N, K);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel6 launch failed: " + std::string(cudaGetErrorString(err)));
    }
}

// kernel6: 16x16x16 MMA指令，带bias
template <typename T>
void advanced_matmul_kernel6_with_bias(const Tensor<T> &A, const Tensor<T> &B, const Tensor<T> &bias, Tensor<T> *C,
                                       cudaStream_t stream) {
    const std::vector<size_t> &A_shape = A.sizes();
    const std::vector<size_t> &B_shape = B.sizes();
    const std::vector<size_t> &bias_shape = bias.sizes();

    size_t M = A_shape[0];
    size_t K = A_shape[1];
    size_t N = B_shape[1];

    if (bias_shape.size() != 1 || bias_shape[0] != N) {
        throw std::runtime_error("Bias dimensions do not match for kernel6");
    }

    // 计算grid和block尺寸
    dim3 threadsPerBlock(WAPR_NUM * 32);
    dim3 numBlocks((M + BM - 1) / BM, (N + BN - 1) / BN);

    // 启动kernel6 with bias
    advanced_kernels::kernel6_with_bias<T, BM, BN, BK, WMMA_M, WMMA_N, WMMA_K, WAPR_NUM, K_STAGE, WARP_TILE_M,
                                        WARP_TILE_N><<<numBlocks, threadsPerBlock, 0, stream>>>(
        A.data_ptr(), B.data_ptr(), bias.data_ptr(), C->data_ptr(), M, N, K);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel6 with bias launch failed: " + std::string(cudaGetErrorString(err)));
    }
}

// 显式实例化模板
template void advanced_matmul_kernel5<float>(const Tensor<float> &A, const Tensor<float> &B, Tensor<float> *C,
                                             cudaStream_t stream);
template void advanced_matmul_kernel5<__nv_bfloat16>(const Tensor<__nv_bfloat16> &A, const Tensor<__nv_bfloat16> &B,
                                                     Tensor<__nv_bfloat16> *C, cudaStream_t stream);

template void advanced_matmul_kernel5_with_bias<float>(const Tensor<float> &A, const Tensor<float> &B,
                                                       const Tensor<float> &bias, Tensor<float> *C,
                                                       cudaStream_t stream);
template void advanced_matmul_kernel5_with_bias<__nv_bfloat16>(const Tensor<__nv_bfloat16> &A,
                                                               const Tensor<__nv_bfloat16> &B,
                                                               const Tensor<__nv_bfloat16> &bias,
                                                               Tensor<__nv_bfloat16> *C, cudaStream_t stream);

template void advanced_matmul_kernel6<float>(const Tensor<float> &A, const Tensor<float> &B, Tensor<float> *C,
                                             cudaStream_t stream);
template void advanced_matmul_kernel6<__nv_bfloat16>(const Tensor<__nv_bfloat16> &A, const Tensor<__nv_bfloat16> &B,
                                                     Tensor<__nv_bfloat16> *C, cudaStream_t stream);

template void advanced_matmul_kernel6_with_bias<float>(const Tensor<float> &A, const Tensor<float> &B,
                                                       const Tensor<float> &bias, Tensor<float> *C,
                                                       cudaStream_t stream);
template void advanced_matmul_kernel6_with_bias<__nv_bfloat16>(const Tensor<__nv_bfloat16> &A,
                                                               const Tensor<__nv_bfloat16> &B,
                                                               const Tensor<__nv_bfloat16> &bias,
                                                               Tensor<__nv_bfloat16> *C, cudaStream_t stream);

}  // namespace cuda_OP