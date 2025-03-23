

#include "cudaOP.cuh"
#include <cstdio> // //printf
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <math.h>
#include <stdexcept>
#include <vector>
// -----------------
// --------------------------------------------------
// softmax 算子实现 (支持多维张量)
// --------------------------------------------------

// 3D softmax 内核（用于 3D 张量，假设 softmax 操作在
// dim==2，即对序列长度进行归一化），支持 mask 逻辑
namespace cuda_OP {
  
__global__ void softmax_3d_kernel(float *data, int seq_len, int n_heads,
                                  int total_seq_len, bool mask, int offset) {
  // 每个 block 负责一行（对应一个 softmax 操作）
  int idx = blockIdx.x;
  int seq_id = idx / n_heads;
  int head_id = idx % n_heads;
  if (seq_id >= seq_len || head_id >= n_heads)
    return;

  int start_idx = seq_id * (n_heads * total_seq_len) + head_id * total_seq_len;
  int valid_length = total_seq_len;
  if (mask) {
    // 将 0-based seq_id 转换为计数（包括当前元素），再结合 kvcache 的 offset
    valid_length = (offset > 0 ? offset + seq_id : seq_id) + 1;
    if (valid_length > total_seq_len)
      valid_length = total_seq_len;
  }

  // 固定共享内存大小为 64 个 float（假定 blockDim.x == 64）
  __shared__ float sdata[64];
  int tid = threadIdx.x;

  // ----- 第一遍归约：计算最大值 -----
  // 每个线程遍历多个元素，计算局部最大值
  float thread_max = -1e9f;
  for (int i = tid; i < total_seq_len; i += blockDim.x) {
    float val = (mask && (i >= valid_length)) ? -1e9f : data[start_idx + i];
    thread_max = fmaxf(thread_max, val);
  }

  // 使用 warp 内归约：在同一 warp 内用 __shfl_down_sync 完成归约
  float local_max = thread_max;
  for (int off = warpSize / 2; off > 0; off /= 2) {
    local_max =
        fmaxf(local_max, __shfl_down_sync(0xFFFFFFFF, local_max, off, 32));
  }
  // 每个 warp 的 lane0 将归约结果写入共享内存
  if ((tid % warpSize) == 0) {
    sdata[tid / warpSize] = local_max;
  }
  __syncthreads();

  // 跨 warp归约：由于 blockDim.x==64，只有 2 个 warp，采用原有共享内存归约
  float max_val;
  if (tid < (blockDim.x / warpSize)) { // 仅 tid==0和tid==1参与
    // 简单归约2个 warp的结果，避免使用 __shfl_down_sync 对不足 warpSize
    // 的归约
    if (blockDim.x / warpSize == 2) {
      if (tid == 0) {
        max_val = sdata[0] > sdata[1] ? sdata[0] : sdata[1];
        sdata[0] = max_val;
      }
    }
    // 若有更多 warp，可采用循环归约（此处仅针对2个 warp）
  }
  __syncthreads();
  max_val = sdata[0]; // 全部线程均可获取全局最大值

  // ----- 第二遍归约：计算指数值和求和 -----
  // 每个线程遍历负责的部分，计算归一化后的指数值，并累加局部和
  float thread_sum = 0.0f;
  for (int i = tid; i < total_seq_len; i += blockDim.x) {
    float val = (mask && (i >= valid_length)) ? -1e9f : data[start_idx + i];
    // 为防止指数爆炸，先减去 max_val
    float exp_val = expf(val - max_val);
    data[start_idx + i] = exp_val;
    thread_sum += exp_val;
  }

  // 使用 warp 内归约求和
  float local_sum = thread_sum;
  for (int off = warpSize / 2; off > 0; off /= 2) {
    local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, off);
  }
  // 每个 warp 的 lane0 将局部和写入共享内存
  if ((tid % warpSize) == 0) {
    sdata[tid / warpSize] = local_sum;
  }
  __syncthreads();

  // 跨 warp归约：对 2 个 warp的局部和进行归约，使用原有共享内存归约方式

  int numWarps = blockDim.x / warpSize;
  if (tid < numWarps) {
    float local_cross = sdata[tid];
    unsigned int active_mask = (1 << numWarps) - 1;
    local_cross += __shfl_down_sync(active_mask, local_cross, 1, numWarps);
    if (tid == 0) {
      sdata[0] = local_cross;
    }
  }
  float sum_val;
  // if (tid < (blockDim.x / warpSize)) {
  //   if (blockDim.x / warpSize == 2) {
  //     if (tid == 0) {
  //       sum_val = sdata[0] + sdata[1];
  //       sdata[0] = sum_val;
  //     }
  //   }
  // }
  __syncthreads();
  sum_val = sdata[0]; // 广播全局求和结果

  // ----- 第三遍：归一化 -----
  // 每个线程对负责的部分进行归一化
  for (int i = tid; i < total_seq_len; i += blockDim.x) {
    data[start_idx + i] /= sum_val;
  }
}

__global__ void softmax_3d_kernel_normal(float *data, int seq_len, int n_heads,
                                         int total_seq_len, bool mask,
                                         int offset) {

  int idx = blockIdx.x;
  int seq_id = idx / n_heads;
  int head_id = idx % n_heads;
  if (seq_id >= seq_len || head_id >= n_heads) {
    return;
  }
  int start_idx = seq_id * (n_heads * total_seq_len) + head_id * total_seq_len;
  int valid_length = total_seq_len;
  if (mask) {
    valid_length = (offset > 0 ? offset + seq_id : seq_id) + 1;
  }
  __shared__ float sdata[64];
  int tid = threadIdx.x;
  float thread_max = -1e9f;

  // 1. 寻找最大值
  for (int i = 0; i < blockDim.x; i += blockDim.x) {
    float val = (mask && (i >= valid_length)) ? -1e9f : data[start_idx + i];
    thread_max = fmaxf(thread_max, val);
  }
  sdata[tid] = thread_max;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] = sdata[tid] > sdata[tid + s] ? sdata[tid] : sdata[tid + s];
    }
    __syncthreads();
  }
  float max_val = sdata[0];
  __syncthreads();
  // 2. 计算expf
  float thread_sum = 0.0f;
  for (int i = tid; i < total_seq_len; i += blockDim.x) {
    float val = (mask && (i >= valid_length)) ? -1e9f : data[start_idx + i];
    float exp_val = expf(val - max_val);
    // 写入全局内存保存结果
    data[start_idx + i] = exp_val;
    thread_sum += exp_val;
  }
  sdata[tid] = thread_sum;
  __syncthreads();

  // 归约计算总和
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }
  float sum_val = sdata[0];
  __syncthreads();

  // 第三遍：归一化
  for (int i = tid; i < total_seq_len; i += blockDim.x) {
    data[start_idx + i] /= sum_val;
  }
}

// CUDA 版 softmax 函数（默认 mask 为 true，可手动传入
// false），要求输出张量与输入张量形状一致
void softmax(Tensor<float> *output, const Tensor<float> *input, int dim,
             bool mask, int offset) {
  // 如果 output 与 input 不同，则先复制数据（设备内拷贝）
  if (output != input) {
    size_t total = 1;
    for (auto s : input->sizes())
      total *= s;
    checkCudaError(cudaMemcpy(output->data_ptr(), input->data_ptr(),
                              total * sizeof(float), cudaMemcpyDeviceToDevice));
  }
  const std::vector<size_t> &shape = input->sizes();
  // 我们对序列长度归一化
  if (shape.size() == 3 && dim == 2) {
    int seq_len = shape[0];
    int n_heads = shape[1];
    int total_seq_len = shape[2];
    int total_rows = seq_len * n_heads;
    int THREADS_PER_BLOCK = 64; // 可根据具体情况调节
    // int shared_mem_size = THREADS_PER_BLOCK * sizeof(float);
    softmax_3d_kernel<<<total_rows, THREADS_PER_BLOCK>>>(
        output->data_ptr(), seq_len, n_heads, total_seq_len, mask, offset);
  } else if (shape.size() == 2 && dim == 1) {
    int seq_len = 1;
    int n_heads = shape[0];
    int total_seq_len = shape[1];
    int total_rows = seq_len * n_heads;
    int THREADS_PER_BLOCK = 64; // 可根据具体情况调节
    // int shared_mem_size = THREADS_PER_BLOCK * sizeof(float);
    softmax_3d_kernel<<<total_rows, THREADS_PER_BLOCK>>>(
        output->data_ptr(), seq_len, n_heads, total_seq_len, mask, offset);
  } else {
    throw std::runtime_error(
        "softmax: Unsupported tensor dimension or dim value");
  }
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaDeviceSynchronize());
}
} // namespace cuda_OP