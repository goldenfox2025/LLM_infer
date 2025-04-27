#ifndef CUDA_GQA_GEMM_CUH
#define CUDA_GQA_GEMM_CUH

#include <cuda_bf16.h> // 提供 nv_bfloat16, __float2bfloat16 等
#include <cuda_fp16.h> // 提供 half, __float2half, __half2float, __hadd, half2 等
#include <stdint.h>    // 提供 int64_t 等标准整数类型

#include <cstdio>    // 提供 fprintf, stderr (用于可能的错误输出)
#include <stdexcept> // 提供 std::runtime_error, std::invalid_argument
#include <string>    // 提供 std::string, std::to_string
#include "cudaOP.cuh"

namespace cuda_OP
{
    /**
     * 支持GQA的批处理GEMM内核
     *
     * 这个内核处理批处理矩阵乘法，支持GQA等场景中的索引映射
     * 每个线程块负责计算一个批次中的一个输出块
     * 使用共享内存优化，提高内存访问效率
     */
    template <typename T, int BLOCK_SIZE = 16>
    __global__ void gqa_gemm_kernel(
        const T *Q,     // 查询矩阵 [batch_size, seq_len, n_q_heads, head_dim]
        const T *K,     // 键矩阵 [batch_size, total_seq_len, n_kv_heads, head_dim]
        T *scores,      // 输出分数矩阵 [batch_size, seq_len, n_q_heads, total_seq_len]
        int seq_len,    // 序列长度
        int head_dim,   // 头维度
        int batch_size, // 批次大小
        int n_q_heads,  // Q头数量
        int n_kv_heads, // KV头数量
        int ratio,      // Q头与KV头的比例 (n_q_heads / n_kv_heads)
        int total_seq_len)
    {

        int batch_idx = blockIdx.z / n_q_heads;
        int q_head_idx = blockIdx.z % n_q_heads;

        // 计算对应的KV头索引
        int kv_head_idx = q_head_idx / ratio;

        // 计算当前线程负责的输出元素索引
        int seq_pos = blockIdx.y * BLOCK_SIZE + threadIdx.y;
        int t_seq_pos = blockIdx.x * BLOCK_SIZE + threadIdx.x;
        int scores_offset = batch_idx * total_seq_len * n_q_heads * seq_len  + seq_pos * total_seq_len * n_q_heads + q_head_idx * total_seq_len + t_seq_pos;
        const T *q = Q + batch_idx * n_q_heads * head_dim * seq_len + seq_pos * head_dim * n_q_heads + q_head_idx * head_dim;
        const T *k = K + batch_idx * n_kv_heads * head_dim * total_seq_len + t_seq_pos * head_dim * n_kv_heads + kv_head_idx * head_dim;
        float sum = 0.0f;
        int tid_t = threadIdx.x;
        int tid = threadIdx.y;
        int tile = (head_dim + BLOCK_SIZE - 1) / BLOCK_SIZE;
        __shared__ float smemQ[16][16];
        __shared__ float smemK[16][16];
        for (int i = 0; i < tile; i++)
        {
            if (tid_t + i * BLOCK_SIZE < head_dim && tid + i * BLOCK_SIZE < head_dim)
            {
                smemQ[tid][tid_t] = static_cast<float>(q[tid_t + i * BLOCK_SIZE]);
                smemK[tid_t][tid] = static_cast<float>(k[tid + i * BLOCK_SIZE]);
            }
            __syncthreads();
            for (int j = 0; j < BLOCK_SIZE; j++)
            {
                sum += smemQ[tid][j] * smemK[tid_t][j];
            }
            __syncthreads();
        }

        sum *= rsqrtf(static_cast<float>(head_dim));

        if (seq_pos < seq_len && t_seq_pos < total_seq_len)
        {
            scores[scores_offset] = static_cast<T>(sum);
        }
    }

    /**
     * 支持GQA的GEMM计算包装函数
     *
     * 这个函数专门处理GQA场景中的矩阵乘法，一个KV头对应多个Q头
     * 本项目不支持batchsize 但算子先尝试适配一下各种情况
     */
    template <typename T>
    void launch_gqa_gemm(
        const Tensor<T> &Q,  // 查询张量 [batch_size, n_q_heads, seq_len, head_dim]
        const Tensor<T> &K,  // 键张量 [batch_size, n_kv_heads, seq_len, head_dim]
        Tensor<T> &scores,   // 输出分数张量 [batch_size, n_q_heads, seq_len, seq_len]
        bool transpose_K,    // 是否转置K (通常为true，用于注意力计算)
        cudaStream_t stream) // CUDA流
    {

        int batch_size = 1;

        // 根据张量维度提取参数
        // 张量布局为 [seq_len, n_heads, head_dim]
        int seq_len = Q.sizes()[0];
        int n_q_heads = Q.sizes()[1];
        int head_dim = Q.sizes()[2];
        int total_seq_len = K.sizes()[0];
        int n_kv_heads = K.sizes()[1];

        // 计算Q头与KV头的比例
        if (n_q_heads % n_kv_heads != 0)
        {
            throw std::runtime_error("n_q_heads must be divisible by n_kv_heads");
        }
        int ratio = n_q_heads / n_kv_heads;

        // 检查scores的形状
        if (scores.sizes()[0] != seq_len || scores.sizes()[1] != n_q_heads ||
            scores.sizes()[2] != total_seq_len)
        {
            throw std::runtime_error("scores tensor shape mismatch");
        }

        // 配置内核启动参数
        constexpr int BLOCK_SIZE = 16; // 可以根据GPU架构调整

        // 计算网格维度
        // x维度：K序列长度的块数
        // y维度：Q序列长度的块数
        // z维度：批次 * Q头数
        dim3 grid(
            (total_seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE, // x: K序列长度的块数
            (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE,       // y: Q序列长度的块数
            batch_size * n_q_heads                         // z: 批次 * Q头数
        );

        // 计算块维度
        // 使用2D块，每个线程处理一个输出元素
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);

        // 启动内核
        // 注意：这里我们传递了所有必要的参数，包括序列长度、头维度、批次大小等
        gqa_gemm_kernel<T, BLOCK_SIZE><<<grid, block, 0, stream>>>(
            Q.data_ptr(),      // 查询矩阵
            K.data_ptr(),      // 键矩阵
            scores.data_ptr(), // 输出分数矩阵
            seq_len,           // Q序列长度
            head_dim,          // 头维度
            batch_size,        // 批次大小
            n_q_heads,         // Q头数量
            n_kv_heads,        // KV头数量
            ratio,             // Q头与KV头的比例
            total_seq_len      // K序列长度
        );

        // 检查CUDA错误
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            throw std::runtime_error("CUDA error in launch_gqa_gemm: " +
                                     std::string(cudaGetErrorString(err)));
        }
    }

    // 显式实例化模板函数，以便在其他文件中使用
    template void launch_gqa_gemm<float>(
        const Tensor<float> &Q, const Tensor<float> &K, Tensor<float> &scores,
        bool transpose_K, cudaStream_t stream);

    template void launch_gqa_gemm<half>(
        const Tensor<half> &Q, const Tensor<half> &K, Tensor<half> &scores,
        bool transpose_K, cudaStream_t stream);

    template void launch_gqa_gemm<nv_bfloat16>(
        const Tensor<nv_bfloat16> &Q, const Tensor<nv_bfloat16> &K, Tensor<nv_bfloat16> &scores,
        bool transpose_K, cudaStream_t stream);

} // namespace cuda_OP

#endif // CUDA_GQA_GEMM_CUH