#include <cublas_v2.h>
#include <cuda_bf16.h>  // 提供 __nv_bfloat16
#include <cuda_runtime.h>
#include <math.h>

#include <cmath>
#include <cstdio>
#include <cstdio>  // //printf
#include <iostream>
#include <stdexcept>
#include <vector>

#include "cudaOP.cuh"

namespace cuda_OP {

__device__ inline float bf16_to_float(__nv_bfloat16 x) {
  // fallback: 手动把 bfloat16 高 16 位当成 float
  unsigned short raw;
  memcpy(&raw, &x, sizeof(raw));  // 把 x 的 16 bits 拷贝到 raw
  unsigned int bits = (static_cast<unsigned int>(raw) << 16);

  float f;
  memcpy(&f, &bits, sizeof(f));  // 转成 float
  return f;
}

__device__ inline __nv_bfloat16 float_to_bf16(float f) {
  // fallback: 把 float 的高 16 bits 当作 bfloat16
  unsigned int bits;
  memcpy(&bits, &f, sizeof(bits));
  unsigned short raw = static_cast<unsigned short>(bits >> 16);

  __nv_bfloat16 h;
  memcpy(&h, &raw, sizeof(h));
  return h;
}
template <typename T>
__device__ inline T my_exp(T x);

// 针对 float
template <>
__device__ inline float my_exp<float>(float x) {
  return expf(x);
}

// 针对 double
template <>
__device__ inline double my_exp<double>(double x) {
  return exp(x);
}

// 针对 nvbf16（__nv_bfloat16）
// CUDA 并没有原生 bf16 exp，需要自己先转 float 做计算，再转回 bf16
template <>
__device__ inline __nv_bfloat16 my_exp<__nv_bfloat16>(__nv_bfloat16 x) {
  float fx = bf16_to_float(x);  // 转成 float
  float ef = expf(fx);          // 用 expf 计算
  return float_to_bf16(ef);     // 转回 bfloat16
}

template <typename T>
__device__ inline T my_fmax(T a, T b);

// 针对 float
template <>
__device__ inline float my_fmax<float>(float a, float b) {
  return fmaxf(a, b);
}

// 针对 double
template <>
__device__ inline double my_fmax<double>(double a, double b) {
  return fmax(a, b);
}

// 同理，bf16 也得自己转 float 再做 fmaxf
template <>
__device__ inline __nv_bfloat16 my_fmax<__nv_bfloat16>(__nv_bfloat16 a,
                                                       __nv_bfloat16 b) {
  float fa = bf16_to_float(a);
  float fb = bf16_to_float(b);
  float fm = fmaxf(fa, fb);
  return float_to_bf16(fm);
}

// -----------------
// --------------------------------------------------
// softmax 算子实现 (支持多维张量)
// --------------------------------------------------

// 3D softmax 内核（用于 3D 张量，假设 softmax 操作在
// dim==2，即对序列长度进行归一化），支持 mask 逻辑
template <typename T>
__global__ void softmax_3d_kernel_v2(T* data, int seq_len, int n_heads,
                                     int total_seq_len, bool mask, int offset) {
  // 每个 block 负责一行（对应一个 softmax 操作）
  int idx = blockIdx.x;
  int seq_id = idx / n_heads;
  int head_id = idx % n_heads;
  if (seq_id >= seq_len || head_id >= n_heads) return;

  int start_idx = seq_id * (n_heads * total_seq_len) + head_id * total_seq_len;
  int valid_length = total_seq_len;
  if (mask) {
    // 将 0-based seq_id 转换为计数（包括当前元素），再结合 kvcache 的 offset
    valid_length = (offset > 0 ? offset + seq_id : seq_id) + 1;
    if (valid_length > total_seq_len) valid_length = total_seq_len;
  }

  // 固定共享内存大小为 64 个 float（假定 blockDim.x == 64）
  __shared__ T sdata[64];
  int tid = threadIdx.x;

  // ----- 第一遍归约：计算最大值 -----
  // 每个线程遍历多个元素，计算局部最大值
  T thread_max = T(-1e9);
  for (int i = tid; i < total_seq_len; i += blockDim.x) {
    T val = (mask && (i >= valid_length)) ? T(-1e9) : data[start_idx + i];
    thread_max = my_fmax(thread_max, val);
  }

  // 使用 warp 内归约：在同一 warp 内用 __shfl_down_sync 完成归约
  T local_max = thread_max;
  for (int off = warpSize / 2; off > 0; off /= 2) {
    local_max =
        my_fmax(local_max, __shfl_down_sync(0xFFFFFFFF, local_max, off, 32));
  }
  // 每个 warp 的 lane0 将归约结果写入共享内存
  if ((tid % warpSize) == 0) {
    sdata[tid / warpSize] = local_max;
  }
  __syncthreads();

  // 跨 warp归约：由于 blockDim.x==64，只有 2 个 warp，采用原有共享内存归约
  T max_val;
  if (tid < (blockDim.x / warpSize)) {
    // 仅 tid==0和tid==1参与
    // 简单归约2个 warp的结果，避免使用 __shfl_down_sync对不足 warpSize 的归约
    if (blockDim.x / warpSize == 2) {
      if (tid == 0) {
        max_val = sdata[0] > sdata[1] ? sdata[0] : sdata[1];
        sdata[0] = max_val;
      }
    }
    // 若有更多 warp，可采用循环归约（此处仅针对2个 warp）
  }
  __syncthreads();
  max_val = sdata[0];  // 全部线程均可获取全局最大值

  // ----- 第二遍归约：计算指数值和求和 -----
  // 每个线程遍历负责的部分，计算归一化后的指数值，并累加局部和
  T thread_sum = T(0);
  for (int i = tid; i < total_seq_len; i += blockDim.x) {
    T val = (mask && (i >= valid_length)) ? T(-1e9) : data[start_idx + i];
    // 为防止指数爆炸，先减去 max_val
    T exp_val = my_exp(val - max_val);
    data[start_idx + i] = exp_val;
    thread_sum += exp_val;
  }

  // 使用 warp 内归约求和
  T local_sum = thread_sum;
  for (int off = warpSize / 2; off > 0; off /= 2) {
    local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, off, 32);
  }
  // 每个 warp 的 lane0 将局部和写入共享内存
  if ((tid % warpSize) == 0) {
    sdata[tid / warpSize] = local_sum;
  }
  __syncthreads();

  // 跨 warp归约：对 2 个 warp的局部和进行归约，使用原有共享内存归约方式
  int numWarps = blockDim.x / warpSize;
  if (tid < numWarps) {
    T local_cross = sdata[tid];
    unsigned int active_mask = (1 << numWarps) - 1;
    local_cross += __shfl_down_sync(active_mask, local_cross, 1, numWarps);
    if (tid == 0) {
      sdata[0] = local_cross;
    }
  }
  T sum_val;
  __syncthreads();
  sum_val = sdata[0];  // 广播全局求和结果

  // ----- 第三遍：归一化 -----
  // 每个线程对负责的部分进行归一化
  for (int i = tid; i < total_seq_len; i += blockDim.x) {
    data[start_idx + i] /= sum_val;
  }
}

template <typename T>
__global__ void softmax_3d_kernel_v2(T* data, int seq_len, int n_heads,
                                     int total_seq_len, bool mask, int offset,
                                     float temperature) {
  // 每个 block 负责一行（对应一个 softmax 操作）
  int idx = blockIdx.x;
  int seq_id = idx / n_heads;
  int head_id = idx % n_heads;
  if (seq_id >= seq_len || head_id >= n_heads) return;

  int start_idx = seq_id * (n_heads * total_seq_len) + head_id * total_seq_len;
  int valid_length = total_seq_len;
  if (mask) {
    // 将 0-based seq_id 转换为计数（包括当前元素），再结合 kvcache 的 offset
    valid_length = (offset > 0 ? offset + seq_id : seq_id) + 1;
    if (valid_length > total_seq_len) valid_length = total_seq_len;
  }

  // 固定共享内存大小为 64 个 float（假定 blockDim.x == 64）
  __shared__ T sdata[64];
  int tid = threadIdx.x;

  // ----- 第一遍归约：计算最大值 -----
  // 每个线程遍历多个元素，计算局部最大值
  T thread_max = T(-1e9);
  for (int i = tid; i < total_seq_len; i += blockDim.x) {
    T val = (mask && (i >= valid_length))
                ? T(-1e9)
                : ((data[start_idx + i]) / T(temperature));
    thread_max = my_fmax(thread_max, val);
  }

  // 使用 warp 内归约：在同一 warp 内用 __shfl_down_sync 完成归约
  T local_max = thread_max;
  for (int off = warpSize / 2; off > 0; off /= 2) {
    local_max =
        my_fmax(local_max, __shfl_down_sync(0xFFFFFFFF, local_max, off, 32));
  }
  // 每个 warp 的 lane0 将归约结果写入共享内存
  if ((tid % warpSize) == 0) {
    sdata[tid / warpSize] = local_max;
  }
  __syncthreads();

  // 跨 warp归约：由于 blockDim.x==64，只有 2 个 warp，采用原有共享内存归约
  T max_val;
  if (tid < (blockDim.x / warpSize)) {
    // 仅 tid==0和tid==1参与
    // 简单归约2个 warp的结果，避免使用 __shfl_down_sync对不足 warpSize 的归约
    if (blockDim.x / warpSize == 2) {
      if (tid == 0) {
        max_val = sdata[0] > sdata[1] ? sdata[0] : sdata[1];
        sdata[0] = max_val;
      }
    }
    // 若有更多 warp，可采用循环归约（此处仅针对2个 warp）
  }
  __syncthreads();
  max_val = sdata[0];  // 全部线程均可获取全局最大值

  // ----- 第二遍归约：计算指数值和求和 -----
  // 每个线程遍历负责的部分，计算归一化后的指数值，并累加局部和
  T thread_sum = T(0);
  for (int i = tid; i < total_seq_len; i += blockDim.x) {
    T val = (mask && (i >= valid_length))
                ? T(-1e9)
                : (data[start_idx + i] / T(temperature));
    // 为防止指数爆炸，先减去 max_val
    T exp_val = my_exp(val - max_val);
    data[start_idx + i] = exp_val;
    thread_sum += exp_val;
  }
  // 使用 warp 内归约求和
  T local_sum = thread_sum;
  for (int off = warpSize / 2; off > 0; off /= 2) {
    local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, off, 32);
  }
  // 每个 warp 的 lane0 将局部和写入共享内存
  if ((tid % warpSize) == 0) {
    sdata[tid / warpSize] = local_sum;
  }
  __syncthreads();

  // 跨 warp归约：对 2 个 warp的局部和进行归约，使用原有共享内存归约方式
  int numWarps = blockDim.x / warpSize;
  if (tid < numWarps) {
    T local_cross = sdata[tid];
    unsigned int active_mask = (1 << numWarps) - 1;
    local_cross += __shfl_down_sync(active_mask, local_cross, 1, numWarps);
    if (tid == 0) {
      sdata[0] = local_cross;
    }
  }
  T sum_val;
  __syncthreads();
  sum_val = sdata[0];  // 广播全局求和结果

  // ----- 第三遍：归一化 -----
  // 每个线程对负责的部分进行归一化
  for (int i = tid; i < total_seq_len; i += blockDim.x) {
    data[start_idx + i] /= sum_val;
  }
}

template <typename T>
__global__ void softmax_3d_kernel_v1(T* data, int seq_len, int n_heads,
                                     int total_seq_len, bool mask, int offset) {
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
  __shared__ T sdata[64];
  int tid = threadIdx.x;
  T thread_max = T(-1e9);
  for (int i = 0; i < total_seq_len; i += blockDim.x) {
    T val = (mask && (i >= valid_length)) ? T(-1e9) : data[start_idx + i];
    thread_max = my_fmax(thread_max, val);
  }
  sdata[tid] = thread_max;
  for (int i = blockDim.x >> 1; i > 0; i >>= 1) {
    if (tid < i) {
      sdata[tid] = my_fmax(sdata[tid], sdata[tid + i]);
    }
    __syncthreads();
  }

  T max_val = sdata[0];
  __syncthreads();
  T thread_sum = T(0);
  for (int i = tid; i < total_seq_len; i += blockDim.x) {
    T val = (mask && i >= valid_length ? T(-1e9) : data[start_idx + i]);
    T exp_val = my_exp(val - max_val);
    data[start_idx + i] = exp_val;
    thread_sum += exp_val;
  }

  sdata[tid] = thread_sum;
  __syncthreads();

  for (int i = blockDim.x >> 1; i > 0; i >>= 1) {
    if (tid < i) {
      sdata[tid] += sdata[tid + i];
    }
    __syncthreads();
  }
  T sum_val = sdata[0];
  for (int i = tid; i < total_seq_len; i += blockDim.x) {
    data[start_idx + i] /= sum_val;
  }
}

// CUDA 版 softmax 函数（默认 mask 为 true，可手动传入
// false），要求输出张量与输入张量形状一致
template <typename T>
void softmax(Tensor<T>* output, const Tensor<T>* input, int dim, bool mask,
             int offset, float temperature) {
  // 如果 output 与 input 不同，则先复制数据（设备内拷贝）
  if (output != input) {
    size_t total = 1;
    for (auto s : input->sizes()) total *= s;
    checkCudaError(cudaMemcpy(output->data_ptr(), input->data_ptr(),
                              total * sizeof(T), cudaMemcpyDeviceToDevice));
  }
  const std::vector<size_t>& shape = input->sizes();
  // 我们对序列长度归一化

  if (shape.size() == 3 && dim == 2) {
    int seq_len = shape[0];
    int n_heads = shape[1];
    int total_seq_len = shape[2];
    int total_rows = seq_len * n_heads;
    int THREADS_PER_BLOCK = 64;  // 可根据具体情况调节
    // int shared_mem_size = THREADS_PER_BLOCK * sizeof(T);
    softmax_3d_kernel_v2<T><<<total_rows, THREADS_PER_BLOCK>>>(
        output->data_ptr(), seq_len, n_heads, total_seq_len, mask, offset);
  } else if (shape.size() == 2 && dim == 1) {
    int seq_len = 1;
    int n_heads = shape[0];
    int total_seq_len = shape[1];
    int total_rows = seq_len * n_heads;
    int THREADS_PER_BLOCK = 64;  // 可根据具体情况调节
    // int shared_mem_size = THREADS_PER_BLOCK * sizeof(T);
    if (temperature >= 0) {
      softmax_3d_kernel_v2<T><<<total_rows, THREADS_PER_BLOCK>>>(
          output->data_ptr(), seq_len, n_heads, total_seq_len, mask, offset,
          temperature);
    } else {
      softmax_3d_kernel_v2<T><<<total_rows, THREADS_PER_BLOCK>>>(
          output->data_ptr(), seq_len, n_heads, total_seq_len, mask, offset);
    }
  } else {
    throw std::runtime_error(
        "softmax: Unsupported tensor dimension or dim value");
  }
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaDeviceSynchronize());
}

template void softmax<nvbf16>(Tensor<nvbf16>*, const Tensor<nvbf16>*, int, bool,
                              int, float);

template void softmax<float>(Tensor<float>*, const Tensor<float>*, int, bool,
                             int, float);

}  // namespace cuda_OP
