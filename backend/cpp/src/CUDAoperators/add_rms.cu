#include <cuda_bf16.h> // Required for nv_bfloat16, nv_bfloat162, __hadd2 (though hadd2 is unused correctly now)
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h> // Required for printf

#include <algorithm> // For std::min
#include <cstdio>
#include <iostream>
#include <stdexcept>   // For std::runtime_error
#include <type_traits> // For std::is_same_v
#include <vector>

// Assume cudaOP.cuh contains Tensor definition, checkCudaError, etc.
#include "cudaOP.cuh"

namespace cuda_OP
{
    // warp_reduce_sum 函数保持不变
    __device__ inline float warp_reduce_sum(float val)
    {
        for (int offset = 32 / 2; offset > 0; offset /= 2)
        {
            val += __shfl_down_sync(__activemask(), val, offset);
        }
        return val;
    }

    template <typename T>
    __global__ void add_rms_kernel(
        T *output,
        T *input,
        const T *add_,
        const T *weight, float eps,
        size_t row_size)
    {
        // 获取当前 block 和 thread 的 ID
        int row = blockIdx.x;
        int tid = threadIdx.x;
        int nthreads = blockDim.x;
        int lane = tid % warpSize;
        int warp_id = tid / warpSize;

        T *in_row = input + row * row_size;
        T *out_row = output + row * row_size;
        const T *add_row = add_ + row * row_size;
        float val[5];
        int flag = 0;
        float local_sum = 0.0f;
        for (size_t i = tid; i < row_size; i += nthreads)
        {


            val[flag++] = static_cast<float>(in_row[i] + add_row[i]);
          

            // 累加平方和
            local_sum += val[flag-1] * val[flag-1];
        }

   
        float warp_sum = warp_reduce_sum(local_sum);


        __shared__ float s_warp_sums[32]; 

     
        if (lane == 0)
        {

            s_warp_sums[warp_id] = warp_sum;
        }
        __syncthreads();

        float block_sum = 0.0f;
        if (warp_id == 0)
        {
            int num_warps_in_block = (nthreads + warpSize - 1) / warpSize;

            float warp_partial_sum = (tid < num_warps_in_block) ? s_warp_sums[tid] : 0.0f;

            // 在 Warp 0 内进行最终的归约
            block_sum = warp_reduce_sum(warp_partial_sum);

            // ---
        }

        __shared__ float s_inv_rms;
        if (tid == 0)
        {
            float mean_sq = block_sum / row_size;
            float rsqrt_arg = mean_sq + eps;
            s_inv_rms = rsqrtf(rsqrt_arg);
        }
        __syncthreads(); // 确保所有线程都能读到 s_inv_rms

        // 5. 应用 RMSNorm: output = (input + add) * inv_rms * weight
        float inv_rms = s_inv_rms; // 所有线程获取计算好的 inv_rms
        flag = 0;
        for (size_t i = tid; i < row_size; i += nthreads)
        {
            if (i < row_size) // 边界检查
            {
                // 重新计算 val = input + add (或者从 shared memory 读取，如果做了优化)
                // float val_f = static_cast<float>(in_row[i]);

                // float add_f = static_cast<float>(add_row[i]);

                // float val = static_cast<float>(in_row[i] + add_row[i]);

                // 读取 weight
                float w = static_cast<float>(weight[i]);

                // 计算最终结果
                float normalized_val = val[flag++] * inv_rms;
                float scaled_val = normalized_val * w;

                out_row[i] = static_cast<T>(scaled_val);
                in_row[i] = static_cast<T>(val[flag-1]); 
            }
        }
    }


    template <typename T>
    void add_rms(Tensor<T> *output,  Tensor<T> *input, const Tensor<T> *add_,
                 const Tensor<T> *weight, float eps, cudaStream_t stream)
    {
        size_t seq_len = input->sizes()[0];
        size_t d = input->sizes()[1]; // row_size

        int threads_per_block = 1024; 
        // 简单的启发式：如果 d 较小，减少线程数以避免浪费
        if (d < 1024)
        {

            if (d <= 32)
                threads_per_block = 32;
            else if (d <= 64)
                threads_per_block = 64;
            else if (d <= 128)
                threads_per_block = 128;
            else if (d <= 256)
                threads_per_block = 256;
            else if (d <= 512)
                threads_per_block = 512;
            else
                threads_per_block = 1024;
        }

        dim3 block_dim(threads_per_block);
        dim3 grid_dim(seq_len);

        // --- 调用 Debug Kernel ---
        add_rms_kernel<T><<<grid_dim, block_dim, 0, stream>>>(
            output->data_ptr(), input->data_ptr(), add_->data_ptr(), weight->data_ptr(), eps, d);
        // ---

        checkCudaError(cudaGetLastError());

    }

    // 模板实例化保持不变
    template void add_rms<float>(Tensor<float> *,  Tensor<float> *, const Tensor<float> *,
                                 const Tensor<float> *, float, cudaStream_t);
    template void add_rms<nvbf16>(Tensor<nvbf16> *,  Tensor<nvbf16> *, const Tensor<nvbf16> *,
                                  const Tensor<nvbf16> *, float, cudaStream_t);

} // namespace cuda_OP