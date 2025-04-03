#include <cublas_v2.h>
#include <cuda_bf16.h>  // 提供 __nv_bfloat16 定义
#include <cuda_runtime.h>
#include <math.h>

#include <cstdio>  // printf
#include <iostream>
#include <stdexcept>
#include <vector>

#include "cudaOP.cuh"

namespace cuda_OP {

// --------------------------------------------------
// 工具函数：检查 CUDA 错误
// --------------------------------------------------
void checkCudaError(cudaError_t error) {
  if (error != cudaSuccess) {
    std::cerr << "CUDA错误: " << cudaGetErrorString(error) << std::endl;
    throw std::runtime_error("CUDA操作失败: " +
                             std::string(cudaGetErrorString(error)));
  }
}

// --------------------------------------------------
// RMSNorm 内核与包装函数（模板化）
// --------------------------------------------------

__device__ inline float warp_reduce_sum(float val) {
  // 注意：__activemask() 会返回当前活跃线程的掩码
  for (int offset = 32 / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(__activemask(), val, offset);
  }
  return val;
}

// v1版本
template <typename T>
__global__ void rms_norm_kernel_v1(const T *input, T *output, const T *weight,
                                   float eps, size_t row_size) {
  // 每个 block 处理一行数据
  int row = blockIdx.x;
  const T *in_row = input + row * row_size;
  T *out_row = output + row * row_size;

  int tid = threadIdx.x;
  int nthreads = blockDim.x;

  float local_sum = 0.0f;
  for (size_t i = tid; i < row_size; i += nthreads) {
    float val = static_cast<float>(in_row[i]);
    local_sum += val * val;
  }

  local_sum = warp_reduce_sum(local_sum);

  // 每个 warp 的第一个线程将归约结果写入共享内存
  // 最多支持 32 个 warp
  __shared__ float shared[32];
  int lane = tid % 32;
  int warp_id = tid / 32;
  if (lane == 0) {
    shared[warp_id] = local_sum;
  }
  __syncthreads();

  float block_sum = 0.0f;
  int num_warps = (nthreads + 32 - 1) / 32;
  if (warp_id == 0) {
    float warp_partial_sum = 0.0f;
    if (tid < num_warps) {
      warp_partial_sum = shared[lane];
    }
    block_sum = warp_reduce_sum(warp_partial_sum);
  }

  // 将最终归约结果通过共享内存广播到所有线程
  // 使用共享内存的第一个元素来存储最终结果
  if (tid == 0) {
    shared[0] = block_sum;
  }
  __syncthreads();

  float final_sum = shared[0];

  float rms = sqrtf(final_sum / row_size + eps);

  // 归一化，每个线程负责处理多个元素
  for (size_t i = tid; i < row_size; i += nthreads) {
    float val = static_cast<float>(in_row[i]);
    float w = static_cast<float>(weight[i]);
    out_row[i] = static_cast<T>((val / rms) * w);
  }
}

// 无印版本
template <typename T>
__global__ void rms_norm_kernel(const T *input, T *output, const T *weight,
                                float eps, size_t row_size) {
  int row = blockIdx.x;  // 每个 block 处理一行
  const T *in_row = input + row * row_size;
  T *out_row = output + row * row_size;
  float sum = 0.0f;
  for (int i = 0; i < row_size; i++) {
    float val = static_cast<float>(in_row[i]);
    sum += val * val;
  }
  float rms = sqrtf(sum / row_size + eps);
  for (int i = 0; i < row_size; i++) {
    float val = static_cast<float>(in_row[i]);
    float w = static_cast<float>(weight[i]);
    out_row[i] = static_cast<T>((val / rms) * w);
  }
}

template <typename T>
void rms_norm(Tensor<T> *output, const Tensor<T> *input,
              const Tensor<T> *weight, float eps) {
  // input/output shape 均为 [seq_len, d]
  size_t seq_len = input->sizes()[0];
  size_t d = input->sizes()[1];
  int threads = 64;
  // 无印版本仅支持单线程
  rms_norm_kernel_v1<<<seq_len, threads>>>(
      input->data_ptr(), output->data_ptr(), weight->data_ptr(), eps, d);
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaDeviceSynchronize());
}

// --------------------------------------------------
// rope 内核与包装函数（模板化）
// --------------------------------------------------

template <typename T>
__global__ void rope_kernel_v1(T *tensor, size_t seq_len, size_t n_heads,
                               size_t head_dim, size_t offset, float theta) {
  // Each thread handles one dimension pair (i, i + head_dim/2) for a specific
  // (seq_idx, head_idx) Grid dimension should be (seq_len * n_heads * head_dim
  // / 2)
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total_rotations = seq_len * n_heads * (head_dim / 2);

  if (idx < total_rotations) {
    size_t head_dim_half = head_dim / 2;
    size_t i = idx % head_dim_half;  // Dimension index (0 to head_dim/2 - 1)
    size_t head_flat_idx =
        idx / head_dim_half;  // Flat index for (seq_idx, head_idx)
    size_t seq_idx = head_flat_idx / n_heads;   // Sequence index
    size_t head_idx = head_flat_idx % n_heads;  // Head index

    size_t head_offset = seq_idx * n_heads * head_dim + head_idx * head_dim;
    T *head_ptr = tensor + head_offset;

    // Calculate frequency and rotation angle
    // Inverse frequency calculation is stable across threads working on the
    // same head/seq pos
    float freq = 1.0f / powf(theta, static_cast<float>(2 * i) /
                                        static_cast<float>(head_dim));
    float val = (static_cast<float>(seq_idx) + offset) * freq;
    float cos_val;
    float sin_val;
    __sincosf(val, &sin_val, &cos_val);  // Compute sin and cos together

    // Load the pair of elements
    // Accesses head_ptr[i] and head_ptr[i + head_dim_half]
    // Consecutive threads access consecutive 'i', improving coalescing for both
    // reads.
    float x0_f = static_cast<float>(head_ptr[i]);
    float x1_f = static_cast<float>(head_ptr[i + head_dim_half]);

    // Perform rotation
    float rotated_x0 = x0_f * cos_val - x1_f * sin_val;
    float rotated_x1 = x0_f * sin_val + x1_f * cos_val;

    // Store the rotated pair back
    // Consecutive threads access consecutive 'i', improving coalescing for
    // writes.
    head_ptr[i] = static_cast<T>(rotated_x0);
    head_ptr[i + head_dim_half] = static_cast<T>(rotated_x1);
  }
}

// --- BF16 Specialization using __nv_bfloat162 ---
// This leverages vector types and intrinsics for BF16

// Check if CUDA version supports __nv_bfloat162 intrinsics (usually >= 11.0)
#if defined(__CUDA_ARCH__) && \
    (__CUDA_ARCH__ >= 800)  // Ampere or later recommended

template <>
__global__ void rope_kernel_v1<__nv_bfloat16>(__nv_bfloat16 *tensor,
                                              size_t seq_len, size_t n_heads,
                                              size_t head_dim, size_t offset,
                                              float theta) {
  // Each thread handles TWO dimension pairs using bfloat162 type
  // Total number of bfloat162 pairs to process per head is head_dim / 4
  // Grid dimension should be (seq_len * n_heads * head_dim / 4)
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t head_dim_half = head_dim / 2;
  size_t pairs_per_head_half =
      head_dim_half / 2;  // Each bfloat162 holds 2 elements
  size_t total_vec_rotations = seq_len * n_heads * pairs_per_head_half;

  if (idx < total_vec_rotations) {
    size_t vec_i = idx % pairs_per_head_half;  // Index of the bfloat162 pair (0
                                               // to pairs_per_head_half - 1)
    size_t head_flat_idx =
        idx / pairs_per_head_half;  // Flat index for (seq_idx, head_idx)
    size_t seq_idx = head_flat_idx / n_heads;   // Sequence index
    size_t head_idx = head_flat_idx % n_heads;  // Head index

    size_t head_offset = seq_idx * n_heads * head_dim + head_idx * head_dim;
    __nv_bfloat16 *head_ptr = tensor + head_offset;

    // Calculate dimension indices for the two elements in the vector
    size_t i0 = vec_i * 2;
    size_t i1 = vec_i * 2 + 1;

    // Calculate frequencies and rotation angles for both elements
    float freq0 = 1.0f / powf(theta, static_cast<float>(2 * i0) /
                                         static_cast<float>(head_dim));
    float freq1 = 1.0f / powf(theta, static_cast<float>(2 * i1) /
                                         static_cast<float>(head_dim));

    float val0 = (static_cast<float>(seq_idx) + offset) * freq0;
    float val1 = (static_cast<float>(seq_idx) + offset) * freq1;

    float cos_val0, sin_val0, cos_val1, sin_val1;
    __sincosf(val0, &sin_val0, &cos_val0);
    __sincosf(val1, &sin_val1, &cos_val1);

    // Load a pair of bfloat162 (4 elements total) using vector load
    // reinterpret_cast is necessary for vectorized loads/stores
    __nv_bfloat162 x0_vec = *reinterpret_cast<__nv_bfloat162 *>(
        &head_ptr[i0]);  // Loads elements at i0, i1
    __nv_bfloat162 x1_vec = *reinterpret_cast<__nv_bfloat162 *>(
        &head_ptr[i0 + head_dim_half]);  // Loads elements at i0+d/2, i1+d/2

    // Convert bf16 vectors to float vectors for calculation
    float2 x0_fvec = __bfloat1622float2(x0_vec);
    float2 x1_fvec = __bfloat1622float2(x1_vec);

    // Pack sin/cos values into float2 for potential vector operations (though
    // used component-wise here)
    float2 cos_vec = make_float2(cos_val0, cos_val1);
    float2 sin_vec = make_float2(sin_val0, sin_val1);

    // Perform rotation using float components (or use bf16 intrinsics if
    // preferred, requires bf16 sin/cos) If using intrinsics, convert cos/sin to
    // bf162 first:
    // __nv_bfloat162 cos_bf16 = __float22bfloat162_rn(cos_vec);
    // __nv_bfloat162 sin_bf16 = __float22bfloat162_rn(sin_vec);
    // __nv_bfloat162 neg_x1_vec = __hnegb2(x1_vec); // Negate x1 for FMA
    // __nv_bfloat162 rotated_x0_bf16 = __hfma2(x0_vec, cos_bf16,
    // __hmul2(neg_x1_vec, sin_bf16));
    // __nv_bfloat162 rotated_x1_bf16 = __hfma2(x0_vec, sin_bf16,
    // __hmul2(x1_vec, cos_bf16));

    // Component-wise rotation using float:
    float rot_x0_f0 = x0_fvec.x * cos_vec.x - x1_fvec.x * sin_vec.x;
    float rot_x0_f1 = x0_fvec.y * cos_vec.y - x1_fvec.y * sin_vec.y;
    float rot_x1_f0 = x0_fvec.x * sin_vec.x + x1_fvec.x * cos_vec.x;
    float rot_x1_f1 = x0_fvec.y * sin_vec.y + x1_fvec.y * cos_vec.y;

    // Convert results back to bfloat162
    __nv_bfloat162 rotated_x0_bf16 =
        __float22bfloat162_rn(make_float2(rot_x0_f0, rot_x0_f1));
    __nv_bfloat162 rotated_x1_bf16 =
        __float22bfloat162_rn(make_float2(rot_x1_f0, rot_x1_f1));

    // Store the rotated pairs back using vector store
    *reinterpret_cast<__nv_bfloat162 *>(&head_ptr[i0]) = rotated_x0_bf16;
    *reinterpret_cast<__nv_bfloat162 *>(&head_ptr[i0 + head_dim_half]) =
        rotated_x1_bf16;
  }
}
#endif  // __CUDA_ARCH__ >= 800

template <typename T>
__global__ void rope_kernel(T *tensor, size_t seq_len, size_t n_heads,
                            size_t head_dim, size_t offset, float theta) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < seq_len * n_heads) {
    size_t seq_idx = idx / n_heads;
    size_t head_idx = idx % n_heads;
    T *head_ptr = tensor + seq_idx * n_heads * head_dim + head_idx * head_dim;
    size_t dim_half = head_dim / 2;
    for (size_t i = 0; i < dim_half; i++) {
      float freq = 1.0f / powf(theta, (2.0f * i) / head_dim);
      float val = (seq_idx + offset) * freq;
      float cos_val = cosf(val);
      float sin_val = sinf(val);
      float x0 = static_cast<float>(head_ptr[i]);
      float x1 = static_cast<float>(head_ptr[i + dim_half]);
      head_ptr[i] = static_cast<T>(x0 * cos_val - x1 * sin_val);
      head_ptr[i + dim_half] = static_cast<T>(x0 * sin_val + x1 * cos_val);
    }
  }
}

template <typename T>
void rope(Tensor<T> *x, size_t offset, float theta, cudaStream_t stream) {
  const auto &sizes = x->sizes();
  if (sizes.size() < 3) {
    throw std::runtime_error("rope: tensor must be at least 3D");
  }
  size_t seq_len = sizes[0];
  size_t n_heads = sizes[1];
  size_t head_dim = sizes[2];

  if (head_dim == 0) return;  // Nothing to do
  if (head_dim % 2 != 0) {
    throw std::runtime_error("rope: head_dim must be even");
  }

  int threads = 256;  // Common block size, can be tuned (128, 512, etc.)
  int blocks = 0;
  void *kernel_ptr = nullptr;

  // Select kernel and grid based on type
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    if (head_dim % 4 != 0) {
      // BF16 kernel requires head_dim to be a multiple of 4 for bfloat162
      // loads/stores Fallback to generic kernel or throw error. Here we throw.
      throw std::runtime_error(
          "rope: BF16 requires head_dim to be a multiple of 4 for optimized "
          "kernel");
    }
    size_t total_vec_rotations =
        seq_len * n_heads *
        (head_dim / 4);  // Each thread handles a bfloat162 pair
    blocks = (total_vec_rotations + threads - 1) / threads;
    kernel_ptr = (void *)rope_kernel_v1<__nv_bfloat16>;
    // std::cout << "Using BF16 Optimized Kernel" << std::endl; // For debugging
  } else
#endif
  {
    // Generic FP32/FP16 path
    size_t total_rotations =
        seq_len * n_heads * (head_dim / 2);  // Each thread handles one pair
    blocks = (total_rotations + threads - 1) / threads;
    kernel_ptr = (void *)rope_kernel_v1<T>;
    // std::cout << "Using Generic Optimized Kernel" << std::endl; // For
    // debugging
  }

  if (blocks == 0 && (seq_len * n_heads * head_dim) > 0) {
    // Handle cases where total rotations might be 0 but tensor isn't empty
    // or very small head_dim resulted in 0 rotations/blocks.
    // If head_dim was 0, we returned earlier. If head_dim is > 0, blocks should
    // be > 0. This calculation should ensure blocks > 0 if work needs to be
    // done. If blocks is still 0, it likely means seq_len or n_heads is 0.
    if (seq_len > 0 && n_heads > 0 && head_dim > 0) {
      blocks = 1;  // Launch at least one block if there's data
    } else {
      return;  // No work to do
    }
  }

  // --- Kernel Launch ---
  // We use a function pointer to avoid repeating the launch code inside
  // if/else. Note: Directly using the kernel function name is usually preferred
  // for type safety, but this shows how to handle it if selecting dynamically.
  if (kernel_ptr == (void *)rope_kernel_v1<T>) {
    rope_kernel_v1<T><<<blocks, threads, 0, stream>>>(
        x->data_ptr(), seq_len, n_heads, head_dim, offset, theta);
  }
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  else if (kernel_ptr == (void *)rope_kernel_v1<__nv_bfloat16>) {
    rope_kernel_v1<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16 *>(
            x->data_ptr()),  // Cast needed if T is bf16
        seq_len, n_heads, head_dim, offset, theta);
  }
#endif
  else {
    throw std::runtime_error("Internal error: No valid kernel selected.");
  }

  // --- Error Checking ---
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "CUDA error after rope kernel launch: "
              << cudaGetErrorString(err) << std::endl;
    throw std::runtime_error("CUDA rope kernel launch failed");
  }
  // Optional synchronization for debugging or if stream is null
  if (stream == nullptr) {
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      std::cerr << "CUDA synchronization error after rope: "
                << cudaGetErrorString(err) << std::endl;
      throw std::runtime_error("CUDA rope synchronization failed");
    }
  }
}

// --------------------------------------------------
// SiLU 内核与包装函数（模板化）
// --------------------------------------------------
template <typename T>
__global__ void silu_kernel(T *data, int total) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total) {
    float x = static_cast<float>(data[idx]);
    data[idx] = static_cast<T>(x / (1.0f + expf(-x)));
  }
}

template <typename T>
void silu(Tensor<T> *output, const Tensor<T> *input) {
  size_t total = 1;
  for (auto s : input->sizes()) total *= s;
  if (output->data_ptr() != input->data_ptr()) {
    checkCudaError(cudaMemcpy(output->data_ptr(), input->data_ptr(),
                              total * sizeof(T), cudaMemcpyDeviceToDevice));
  }
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  silu_kernel<<<blocks, threads>>>(output->data_ptr(), total);
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaDeviceSynchronize());
}

// --------------------------------------------------
// 注意力计算：decode 版本 — 计算注意力分数（模板化）
// --------------------------------------------------
template <typename T>
__global__ void attention_scores_kernel(const T *Q, int n_q_h, int dqkv,
                                        const T *K, int cache_length,
                                        T *att_scores, int n_kv_h,
                                        bool is_3d_tensor) {
  int q = blockIdx.x;     // 查询头索引，每个 block 负责一个头
  int pos = threadIdx.x;  // 缓存位置索引
  if (q < n_q_h && pos < cache_length) {
    int n_groups = n_q_h / n_kv_h;  // 头分组数
    int kv_head = q / n_groups;     // 对应的 KV 头
    float dot = 0.0f;
    for (int i = 0; i < dqkv; i++) {
      float q_val;
      if (is_3d_tensor) {
        q_val = static_cast<float>(Q[(0 * n_q_h + q) * dqkv + i]);
      } else {
        q_val = static_cast<float>(Q[q * dqkv + i]);
      }
      float k_val = static_cast<float>(K[(pos * n_kv_h + kv_head) * dqkv + i]);
      dot += q_val * k_val;
    }
    att_scores[q * cache_length + pos] =
        static_cast<T>(dot / sqrtf((float)dqkv));
  }
}

template <typename T>
void compute_attention_scores(const Tensor<T> &Q, const Tensor<T> &K,
                              size_t n_q_h, size_t dqkv, Tensor<T> &att_scores,
                              size_t n_kv_h) {
  size_t cache_length = K.sizes()[0];
  const std::vector<size_t> &Q_shape = Q.sizes();
  bool is_3d_q = (Q_shape.size() == 3);
  if (is_3d_q) {
    if (Q_shape[0] != 1 || Q_shape[1] != n_q_h || Q_shape[2] != dqkv) {
      throw std::runtime_error("Q tensor dimension mismatch");
    }
  } else {
    if (Q_shape[0] != n_q_h || Q_shape[1] != dqkv) {
      throw std::runtime_error("Q tensor dimension mismatch");
    }
  }
  if (K.sizes()[0] != cache_length || K.sizes()[1] != n_kv_h ||
      K.sizes()[2] != dqkv) {
    throw std::runtime_error("K tensor dimension mismatch");
  }
  if (att_scores.sizes()[0] != n_q_h || att_scores.sizes()[1] != cache_length) {
    throw std::runtime_error("attention scores tensor shape mismatch");
  }
  int blocks = n_q_h;
  int threads = std::min(static_cast<int>(cache_length), 1024);
  attention_scores_kernel<<<blocks, threads>>>(
      Q.data_ptr(), n_q_h, dqkv, K.data_ptr(), cache_length,
      att_scores.data_ptr(), n_kv_h, is_3d_q);
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaDeviceSynchronize());
}

// --------------------------------------------------
// 注意力计算：decode 版本 — 计算注意力输出（模板化）
// --------------------------------------------------
template <typename T>
__global__ void att_output_kernel(const T *att_probs, int n_q_h,
                                  int cache_length, int dqkv, const T *V,
                                  T *att_output, int n_kv_h) {
  int q = blockIdx.x;
  int d = blockIdx.y;
  if (q < n_q_h && d < dqkv) {
    int n_groups = n_q_h / n_kv_h;
    int kv_head = q / n_groups;
    float sum = 0.0f;
    for (int pos = 0; pos < cache_length; ++pos) {
      float prob = static_cast<float>(att_probs[q * cache_length + pos]);
      float val = static_cast<float>(V[(pos * n_kv_h + kv_head) * dqkv + d]);
      sum += prob * val;
    }
    att_output[q * dqkv + d] = static_cast<T>(sum);
  }
}

template <typename T>
void compute_att_output(const Tensor<T> &att_probs, const Tensor<T> &V,
                        size_t n_q_h, size_t dqkv, Tensor<T> &att_output,
                        size_t n_kv_h) {
  size_t cache_length = V.sizes()[0];
  if (att_probs.sizes()[0] != n_q_h || att_probs.sizes()[1] != cache_length) {
    throw std::runtime_error("attention probabilities tensor shape mismatch");
  }
  if (V.sizes()[0] != cache_length || V.sizes()[1] != n_kv_h ||
      V.sizes()[2] != dqkv) {
    throw std::runtime_error("V tensor dimension mismatch");
  }
  if (att_output.sizes()[0] != n_q_h || att_output.sizes()[1] != dqkv) {
    throw std::runtime_error("attention output tensor shape mismatch");
  }
  dim3 grid(n_q_h, dqkv);
  att_output_kernel<<<grid, 1>>>(
      att_probs.data_ptr(), static_cast<int>(n_q_h),
      static_cast<int>(cache_length), static_cast<int>(dqkv), V.data_ptr(),
      att_output.data_ptr(), static_cast<int>(n_kv_h));
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaDeviceSynchronize());
}

// --------------------------------------------------
// 注意力计算：prefill 版本 — 计算注意力分数（模板化）
// --------------------------------------------------
template <typename T>
__global__ void attention_scores_prefill_kernel(const T *Q, const T *K,
                                                T *att_scores, int n_q,
                                                int cache_length, int dqkv,
                                                int n_q_h, int n_kv_h) {
  int q = blockIdx.x;   // 扁平化后的查询索引：q in [0, n_q)
  int j = threadIdx.x;  // 缓存位置索引
  if (q < n_q && j < cache_length) {
    int s = q / n_q_h;
    int qh = q % n_q_h;
    int n_groups = n_q_h / n_kv_h;
    int kv_head = qh / n_groups;
    float dot = 0.0f;
    for (int d = 0; d < dqkv; d++) {
      float q_val = static_cast<float>(Q[(s * n_q_h + qh) * dqkv + d]);
      float k_val = static_cast<float>(K[(j * n_kv_h + kv_head) * dqkv + d]);
      dot += q_val * k_val;
    }
    int out_idx = s * (n_q_h * cache_length) + qh * cache_length + j;
    att_scores[out_idx] = static_cast<T>(dot / sqrtf((float)dqkv));
  }
}

template <typename T>
void compute_attention_scores_prefill(const Tensor<T> &Q, const Tensor<T> &K,
                                      Tensor<T> &att_scores, size_t dqkv) {
  // Q [seq_len, n_q_h, dqkv]
  // K [cache_length, n_kv_h, dqkv]
  // att_scores [seq_len, n_q_h, cache_length]
  size_t seq_len = Q.sizes()[0];
  size_t n_q_h = Q.sizes()[1];
  size_t cache_length = K.sizes()[0];
  size_t n_kv_h = K.sizes()[1];

  if (Q.sizes()[2] != dqkv) {
    throw std::runtime_error("Q tensor dimension mismatch");
  }
  if (att_scores.sizes()[0] != seq_len || att_scores.sizes()[1] != n_q_h ||
      att_scores.sizes()[2] != cache_length) {
    throw std::runtime_error("attention scores tensor shape mismatch...");
  }
  if (K.sizes()[2] != dqkv) {
    throw std::runtime_error("K tensor dimension mismatch");
  }
  size_t total_q = seq_len * n_q_h;
  int threads = std::min(static_cast<int>(cache_length), 4096);
  int blocks = static_cast<int>(total_q);
  attention_scores_prefill_kernel<<<blocks, threads>>>(
      Q.data_ptr(), K.data_ptr(), att_scores.data_ptr(),
      static_cast<int>(total_q), static_cast<int>(cache_length),
      static_cast<int>(dqkv), static_cast<int>(n_q_h),
      static_cast<int>(n_kv_h));
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaDeviceSynchronize());
}

// --------------------------------------------------
// 注意力计算：prefill 版本 — 计算注意力输出（模板化）
// --------------------------------------------------
template <typename T>
__global__ void att_output_prefill_kernel(const T *att_probs, const T *V,
                                          T *att_output, int n_q,
                                          int cache_length, int dqkv,
                                          int n_kv_h, int n_q_h) {
  int q = blockIdx.x;   // 查询索引
  int d = threadIdx.x;  // 维度索引
  if (q < n_q && d < dqkv) {
    int s = q / n_q_h;   // 序列索引 (批次索引)
    int qh = q % n_q_h;  // 查询头索引
    int n_groups = n_q_h / n_kv_h;
    int kv_head = qh / n_groups;
    float sum = 0.0f;
    for (int j = 0; j < cache_length; j++) {
      int att_idx = s * (n_q_h * cache_length) + qh * cache_length + j;
      float prob = static_cast<float>(att_probs[att_idx]);
      float val = static_cast<float>(V[(j * n_kv_h + kv_head) * dqkv + d]);
      sum += prob * val;
    }
    int out_idx = (s * n_q_h + qh) * dqkv + d;
    att_output[out_idx] = static_cast<T>(sum);
  }
}

template <typename T>
void compute_att_output_prefill(const Tensor<T> &att_probs, const Tensor<T> &V,
                                Tensor<T> &att_output, size_t n_q_h,
                                size_t dqkv, size_t total_seq_len,
                                size_t n_kv_h) {
  // att_probs: [batch_size, n_q_h, cache_length]
  // V: [cache_length, n_kv_h, dqkv]
  // att_output: [batch_size, n_q_h, dqkv]
  size_t batch_size = att_probs.sizes()[0];
  size_t n_q_h_int = n_q_h;
  size_t cache_length = att_probs.sizes()[2];

  if (att_probs.sizes()[1] != n_q_h_int) {
    throw std::runtime_error("attention probabilities head dimension mismatch");
  }
  if (V.sizes()[0] != cache_length || V.sizes()[1] != n_kv_h ||
      V.sizes()[2] != dqkv) {
    throw std::runtime_error("V tensor dimension mismatch");
  }
  if (att_output.sizes()[0] != batch_size ||
      att_output.sizes()[1] != n_q_h_int || att_output.sizes()[2] != dqkv) {
    throw std::runtime_error("attention output tensor shape mismatch");
  }
  size_t total_q = batch_size * n_q_h_int;
  int threads = std::min(static_cast<int>(dqkv), 1024);
  int blocks = static_cast<int>(total_q);
  att_output_prefill_kernel<<<blocks, threads>>>(
      att_probs.data_ptr(), V.data_ptr(), att_output.data_ptr(),
      static_cast<int>(total_q), static_cast<int>(cache_length),
      static_cast<int>(dqkv), static_cast<int>(n_kv_h),
      static_cast<int>(n_q_h_int));
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaDeviceSynchronize());
}

__global__ void init_curand_state_kernel(curandState *states,
                                         unsigned long long seed,
                                         unsigned long long offset) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    curand_init(seed, 0, offset, &states[0]);
  }
}
void init_curand(curandState *d_states, unsigned long long seed, int offset) {
  int blocks = 1;
  int threads = 1;
  init_curand_state_kernel<<<blocks, threads>>>(d_states, seed, offset);
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaDeviceSynchronize());
}

template void rope<float>(Tensor<float> *, size_t, float, cudaStream_t);
template void rms_norm<float>(Tensor<float> *, const Tensor<float> *,
                              const Tensor<float> *, float);
template void silu<float>(Tensor<float> *, const Tensor<float> *);

template void compute_attention_scores<float>(const Tensor<float> &,
                                              const Tensor<float> &, size_t,
                                              size_t, Tensor<float> &, size_t);
template void compute_att_output<float>(const Tensor<float> &,
                                        const Tensor<float> &, size_t, size_t,
                                        Tensor<float> &, size_t);
template void compute_attention_scores_prefill<float>(const Tensor<float> &,
                                                      const Tensor<float> &,
                                                      Tensor<float> &, size_t);
template void compute_att_output_prefill<float>(const Tensor<float> &,
                                                const Tensor<float> &,
                                                Tensor<float> &, size_t, size_t,
                                                size_t, size_t);

// 对 nvbf16 类型的实例化

template void rope<nvbf16>(Tensor<nvbf16> *, size_t, float, cudaStream_t);
template void rms_norm<nvbf16>(Tensor<nvbf16> *, const Tensor<nvbf16> *,
                               const Tensor<nvbf16> *, float);
template void silu<nvbf16>(Tensor<nvbf16> *, const Tensor<nvbf16> *);

template void compute_attention_scores<nvbf16>(const Tensor<nvbf16> &,
                                               const Tensor<nvbf16> &, size_t,
                                               size_t, Tensor<nvbf16> &,
                                               size_t);
template void compute_att_output<nvbf16>(const Tensor<nvbf16> &,
                                         const Tensor<nvbf16> &, size_t, size_t,
                                         Tensor<nvbf16> &, size_t);
template void compute_attention_scores_prefill<nvbf16>(const Tensor<nvbf16> &,
                                                       const Tensor<nvbf16> &,
                                                       Tensor<nvbf16> &,
                                                       size_t);
template void compute_att_output_prefill<nvbf16>(const Tensor<nvbf16> &,
                                                 const Tensor<nvbf16> &,
                                                 Tensor<nvbf16> &, size_t,
                                                 size_t, size_t, size_t);
}  // namespace cuda_OP
