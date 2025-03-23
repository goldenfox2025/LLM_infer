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

// ==================================================
// 模板化内核及包装函数
// ==================================================

// --------------------------------------------------
// RMSNorm 内核与包装函数（模板化）
// --------------------------------------------------
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
  rms_norm_kernel<<<seq_len, 1>>>(input->data_ptr(), output->data_ptr(),
                                  weight->data_ptr(), eps, d);
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaDeviceSynchronize());
}

// --------------------------------------------------
// rope 内核与包装函数（模板化）
// --------------------------------------------------
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
void rope(Tensor<T> *x, size_t offset, float theta) {
  const auto &sizes = x->sizes();
  if (sizes.size() < 3) {
    throw std::runtime_error("rope: tensor must be at least 3D");
  }
  size_t seq_len = sizes[0];
  size_t n_heads = sizes[1];
  size_t head_dim = sizes[2];
  size_t total_elements = seq_len * n_heads;
  int threads = 256;
  int blocks = (total_elements + threads - 1) / threads;
  rope_kernel<<<blocks, threads>>>(x->data_ptr(), seq_len, n_heads, head_dim,
                                   offset, theta);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    throw std::runtime_error("CUDA rope kernel launch failed");
  }
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::cerr << "CUDA synchronization error: " << cudaGetErrorString(err)
              << std::endl;
    throw std::runtime_error("CUDA rope synchronization failed");
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

template void rope<float>(Tensor<float> *, size_t, float);
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

template void rope<nvbf16>(Tensor<nvbf16> *, size_t, float);
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
