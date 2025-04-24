#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <cmath>
#include <cstdio>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "cudaOP.cuh"

namespace cuda_OP
{

  template <typename T>
  __global__ void softmax_kernel(T *data, int seq_len, int n_heads,
                                 int total_seq_len, bool mask, int offset)
  {
    int idx = blockIdx.x;
    int seq_id = idx / n_heads;
    int head_id = idx % n_heads;
    if (seq_id >= seq_len || head_id >= n_heads)
      return;

    int start_idx = seq_id * (n_heads * total_seq_len) + head_id * total_seq_len;
    int valid_length = total_seq_len;
    if (mask)
    {
      valid_length = (offset > 0 ? offset + seq_id : seq_id) + 1;
      if (valid_length > total_seq_len)
        valid_length = total_seq_len;
    }

    // 动态分配共享内存，大小为线程块大小的一个warp（32个线程）
    extern __shared__ float sdata[];
    int tid = threadIdx.x;

    float thread_max = float(-1e9);
    for (int i = tid; i < total_seq_len; i += blockDim.x)
    {
      float val = (mask && (i >= valid_length)) ? float(-1e9) : float(data[start_idx + i]);
      thread_max = fmaxf(thread_max, val);
    }

    for (int off = warpSize / 2; off > 0; off /= 2)
    {
      thread_max =
          fmaxf(thread_max, __shfl_down_sync(0xFFFFFFFF, thread_max, off, 32));
    }

    if ((tid % warpSize) == 0)
    {
      sdata[tid / warpSize] = thread_max;
    }
    __syncthreads();

    // 跨warp归约计算最大值
    float max_val;
    if (tid < 32) {
      // 只使用第一个warp进行最终归约
      float warp_max = tid < (blockDim.x / warpSize) ? sdata[tid] : float(-1e9);

      // 使用warp内shuffle操作进行归约
      for (int offset = 16; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(0xFFFFFFFF, warp_max, offset);
        warp_max = fmaxf(warp_max, other);
      }

      // 线程0写入最终结果
      if (tid == 0) {
        sdata[0] = warp_max;
      }
    }
    __syncthreads();
    max_val = sdata[0];

    float thread_sum = float(0);
    for (int i = tid; i < total_seq_len; i += blockDim.x)
    {
      float val = (mask && (i >= valid_length)) ? T(-1e9) : data[start_idx + i];
      float exp_val = __expf(val - max_val);
      data[start_idx + i] = exp_val;
      thread_sum += exp_val;
    }

    for (int off = warpSize / 2; off > 0; off /= 2)
    {
      thread_sum += __shfl_down_sync(0xFFFFFFFF, thread_sum, off);
    }

    if ((tid % warpSize) == 0)
    {
      sdata[tid / warpSize] = thread_sum;
    }
    __syncthreads();

    // 跨warp归约计算总和
    if (tid < 32) {
      // 只使用第一个warp进行最终归约
      float warp_sum = tid < (blockDim.x / warpSize) ? sdata[tid] : 0.0f;

      // 使用warp内shuffle操作进行归约
      for (int offset = 16; offset > 0; offset /= 2) {
        warp_sum += __shfl_down_sync(0xFFFFFFFF, warp_sum, offset);
      }

      // 线程0写入最终结果
      if (tid == 0) {
        sdata[0] = warp_sum;
      }
    }
    T sum_val;
    __syncthreads();
    sum_val = static_cast<T>(sdata[0]);

    for (int i = tid; i < total_seq_len; i += blockDim.x)
    {
      data[start_idx + i] /= sum_val;
    }
  }

  template <typename T>
  void softmax(Tensor<T> *output, const Tensor<T> *input, int dim, bool mask,
               int offset, cudaStream_t stream)
  {
    // 如果 output 与 input 不同，则先复制数据（设备内拷贝）
    if (output != input)
    {
      size_t total = 1;
      for (auto s : input->sizes())
        total *= s;
      checkCudaError(cudaMemcpy(output->data_ptr(), input->data_ptr(),
                                total * sizeof(T), cudaMemcpyDeviceToDevice));
    }
    const std::vector<size_t> &shape = input->sizes();
    if (shape.size() == 3 && dim == 2)
    {
      int seq_len = shape[0];
      int n_heads = shape[1];
      int total_seq_len = shape[2];
      int total_rows = seq_len * n_heads;
      int THREADS_PER_BLOCK = 256; // 增加线程数以处理更长的序列
      // 计算每个block需要的共享内存大小：每个warp一个float
      int sharedMemSize = (THREADS_PER_BLOCK / 32 + 1) * sizeof(float);
      softmax_kernel<T><<<total_rows, THREADS_PER_BLOCK, sharedMemSize, stream>>>(
          output->data_ptr(), seq_len, n_heads, total_seq_len, mask, offset);
    }
    else if (shape.size() == 2 && dim == 1)
    {
      int seq_len = 1;
      int n_heads = shape[0];
      int total_seq_len = shape[1];
      int total_rows = seq_len * n_heads;
      int THREADS_PER_BLOCK = 256; // 增加线程数以处理更长的序列
      // 计算每个block需要的共享内存大小：每个warp一个float
      int sharedMemSize = (THREADS_PER_BLOCK / 32 + 1) * sizeof(float);
      softmax_kernel<T><<<total_rows, THREADS_PER_BLOCK, sharedMemSize, stream>>>(
          output->data_ptr(), seq_len, n_heads, total_seq_len, mask, offset);
    }
    else
    {
      throw std::runtime_error(
          "softmax: Unsupported tensor dimension or dim value");
    }
    checkCudaError(cudaGetLastError());
    // checkCudaError(cudaDeviceSynchronize());
  }

  template void softmax<nvbf16>(Tensor<nvbf16> *, const Tensor<nvbf16> *, int, bool,
                                int, cudaStream_t);

  template void softmax<float>(Tensor<float> *, const Tensor<float> *, int, bool,
                               int, cudaStream_t);

} // namespace cuda_OP
