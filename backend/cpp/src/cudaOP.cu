#include "cudaOP.cuh"
#include <cstdio> // //printf
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <math.h>
#include <stdexcept>
#include <vector>

namespace cuda_OP {

// --------------------------------------------------
// 内联函数：检查 CUDA 错误
// --------------------------------------------------
void checkCudaError(cudaError_t error) {
  if (error != cudaSuccess) {
    std::cerr << "CUDA错误: " << cudaGetErrorString(error) << std::endl;
    throw std::runtime_error("CUDA操作失败: " +
                             std::string(cudaGetErrorString(error)));
  }
}

// --------------------------------------------------
// gather 算子实现
// --------------------------------------------------
// kernel：output[i, :] = embedding_table[input[i], :]
// CUDA kernel 使用二维线程块与网格

// embedding_table: [vocab_size, embed_dim]
// input: [seq_len] (里面是 token_id)
// output: [seq_len, embed_dim]
__global__ void gather_kernel(const uint32_t *input,
                              const float *embedding_table, float *output,
                              int seq_len, int embed_dim, int vocab_size) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int total = seq_len * embed_dim;
  for (int idx = tid; idx < total; idx += stride) {
    int row = idx / embed_dim;
    int col = idx % embed_dim;
    if (row >= seq_len || col >= embed_dim) {
      continue;
    }
    uint32_t token_id = input[row];
    if (token_id >= vocab_size) {
      continue;
    }
    int emb_index = token_id * embed_dim + col;
    if (emb_index >= vocab_size * embed_dim) {
      continue;
    }

    float value = embedding_table[emb_index];
    output[idx] = value;
  }
}

void gather(Tensor<float> *output, const Tensor<uint32_t> *input,
            const Tensor<float> *embedding_table) {
  int seq_len = static_cast<int>(input->numel());
  int embed_dim = static_cast<int>(output->sizes()[1]);
  int vocab_size = static_cast<int>(embedding_table->sizes()[0]);

  // 验证设备一致性
  if (input->device() != Device::CUDA ||
      embedding_table->device() != Device::CUDA ||
      output->device() != Device::CUDA) {
    throw std::runtime_error("All tensors must be on CUDA device");
  }

  // 验证输入数据有效性
  if (seq_len <= 0 || embed_dim <= 0 || vocab_size <= 0) {
    throw std::runtime_error("Invalid dimensions in gather");
  }

  int threadsPerBlock = 256;
  int total = seq_len * embed_dim;
  int blocks = (total + threadsPerBlock - 1) / threadsPerBlock;

  gather_kernel<<<blocks, threadsPerBlock>>>(
      input->data_ptr(), embedding_table->data_ptr(), output->data_ptr(),
      seq_len, embed_dim, vocab_size);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error("CUDA gather kernel launch failed");
  }

  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    throw std::runtime_error("CUDA gather synchronization failed");
  }
}

// --------------------------------------------------
// RMSNorm 算子实现
// --------------------------------------------------
__global__ void rms_norm_kernel(const float *input, float *output,
                                const float *weight, float eps,
                                size_t row_size) {
  int row = blockIdx.x; // 每个 block 处理一行
  const float *in_row = input + row * row_size;
  float *out_row = output + row * row_size;
  float sum = 0.0f;
  for (int i = 0; i < row_size; i++) {
    float val = in_row[i];
    sum += val * val;
  }
  float rms = sqrtf(sum / row_size + eps);
  for (int i = 0; i < row_size; i++) {
    out_row[i] = (in_row[i] / rms) * weight[i];
  }
}

void rms_norm(Tensor<float> *output, const Tensor<float> *input,
              const Tensor<float> *weight, float eps) {
  // input/output shape 均为 [seq_len, d]
  size_t seq_len = input->sizes()[0];
  size_t d = input->sizes()[1];
  rms_norm_kernel<<<seq_len, 1>>>(input->data_ptr(), output->data_ptr(),
                                  weight->data_ptr(), eps, d);
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaDeviceSynchronize());
}

// --------------------------------------------------
// matmul 算子实现
// --------------------------------------------------
__global__ void matmul_kernel(const float *A, const float *B, float *C, int M,
                              int K, int N) {
  __shared__ float As[16][16];
  __shared__ float Bs[16][16];
  int row = blockIdx.y * 16 + threadIdx.y;
  int col = blockIdx.x * 16 + threadIdx.x;
  float sum = 0.0f;
  for (int t = 0; t < (16 + K - 1) / 16; ++t) {
    int A_col = t * 16 + threadIdx.x;
    if (row < M && A_col < K) {
      As[threadIdx.y][threadIdx.x] = A[row * K + A_col];
    } else {
      As[threadIdx.y][threadIdx.x] = 0.0f;
    }
    int B_row = t * 16 + threadIdx.y;
    if (col < N && B_row < K) {
      Bs[threadIdx.y][threadIdx.x] = B[col * K + B_row];
    } else {
      Bs[threadIdx.y][threadIdx.x] = 0.0f;
    }
    __syncthreads();
    for (int k = 0; k < 16; ++k) {
      sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    }
    __syncthreads();
  }

  if (row < M && col < N) {
    C[row * N + col] = sum;
  }
}

Tensor<float> matmul(const Tensor<float> &A, const Tensor<float> &B,
                     cudaStream_t stream) {
  const std::vector<size_t> &A_shape = A.sizes();
  const std::vector<size_t> &B_shape = B.sizes();

  size_t m = A_shape[0];
  size_t k = A_shape[1];
  size_t n = B_shape[1];
  Tensor<float> C({m, n}, Device::CUDA);

  // 启动 CUDA 核

  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (m + threadsPerBlock.y - 1) / threadsPerBlock.y);

  // 计算矩阵乘法 C = A * B^T
  matmul_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(
      A.data_ptr(), B.data_ptr(), C.data_ptr(), m, k, n);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error("CUDA kernel launch failed: " +
                             std::string(cudaGetErrorString(err)));
  }

  return C;
}

// --------------------------------------------------
// rope 算子实现
// --------------------------------------------------
__global__ void rope_kernel(float *tensor, size_t seq_len, size_t n_heads,
                            size_t head_dim, size_t offset, float theta) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < seq_len * n_heads) {
    // 计算 seq_idx 和 head_idx
    size_t seq_idx = idx / n_heads;
    size_t head_idx = idx % n_heads;

    // 计算 head_ptr 的起始地址
    float *head_ptr =
        tensor + seq_idx * n_heads * head_dim + head_idx * head_dim;

    // 每个线程处理半个维度
    size_t dim_half = head_dim / 2;
    for (size_t i = 0; i < dim_half; i++) {
      float freq = 1.0f / powf(theta, (2.0f * i) / head_dim);
      float val = (seq_idx + offset) * freq;
      float cos_val = cosf(val);
      float sin_val = sinf(val);

      // 获取需要旋转的两个元素
      float x0 = head_ptr[i];
      float x1 = head_ptr[i + dim_half];

      // 应用旋转
      head_ptr[i] = x0 * cos_val - x1 * sin_val;
      head_ptr[i + dim_half] = x0 * sin_val + x1 * cos_val;
    }
  }
}

void rope(Tensor<float> *x, size_t offset, float theta) {
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

  // 调用 CUDA 核心
  rope_kernel<<<blocks, threads>>>(x->data_ptr(), seq_len, n_heads, head_dim,
                                   offset, theta);

  // 错误检查
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    throw std::runtime_error("CUDA rope kernel launch failed");
  }

  // 同步
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::cerr << "CUDA synchronization error: " << cudaGetErrorString(err)
              << std::endl;
    throw std::runtime_error("CUDA rope synchronization failed");
  }
}

// --------------------------------------------------
// softmax 算子实现 (支持多维张量)
// --------------------------------------------------

// 3D softmax 内核（用于 3D 张量，假设 softmax 操作在
// dim==2，即对序列长度进行归一化），支持 mask 逻辑
__global__ void softmax_3d_kernel(float *data, int seq_len, int n_heads,
                                  int total_seq_len, bool mask, int offset) {
  int idx = blockIdx.x;
  int seq_id = idx / n_heads;
  int head_id = idx % n_heads;
  if (seq_id < seq_len && head_id < n_heads) {
    int start_idx =
        seq_id * (n_heads * total_seq_len) + head_id * total_seq_len;
    int valid_length = total_seq_len;
    if (mask) {
      int query_index = idx / n_heads;
      valid_length = (offset > 0 ? offset + query_index : query_index) + 1;
      if (valid_length > total_seq_len) {
        valid_length = total_seq_len;
      }
    }
    float max_val = -1e9f;
    for (int i = 0; i < total_seq_len; i++) {
      float val = (mask && (i >= valid_length)) ? -1e9f : data[start_idx + i];
      if (val > max_val)
        max_val = val;
    }

    float sum = 0.0f;

    for (int i = 0; i < total_seq_len; i++) {
      float val = (mask && (i >= valid_length)) ? -1e9f : data[start_idx + i];
      float exp_val = expf(val - max_val);
      data[start_idx + i] = exp_val;
      sum += exp_val;
    }
    for (int i = 0; i < total_seq_len; i++) {
      data[start_idx + i] /= sum;
    }
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
    softmax_3d_kernel<<<total_rows, 1>>>(output->data_ptr(), seq_len, n_heads,
                                         total_seq_len, mask, offset);
  } else if (shape.size() == 2 && dim == 1) {
    int seq_len = 1;
    int n_heads = shape[0];
    int total_seq_len = shape[1];
    int total_rows = seq_len * n_heads;
    softmax_3d_kernel<<<total_rows, 1>>>(output->data_ptr(), seq_len, n_heads,
                                         total_seq_len, mask, offset);
  } else {
    throw std::runtime_error(
        "softmax: Unsupported tensor dimension or dim value");
  }
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaDeviceSynchronize());
}

// --------------------------------------------------
// SiLU 算子实现（逐元素）
// --------------------------------------------------
__global__ void silu_kernel(float *data, int total) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total) {
    float x = data[idx];
    data[idx] = x / (1.0f + expf(-x));
  }
}

void silu(Tensor<float> *output, Tensor<float> *input) {
  size_t total = 1;
  for (auto s : input->sizes())
    total *= s;
  if (output->data_ptr() != input->data_ptr()) {
    checkCudaError(cudaMemcpy(output->data_ptr(), input->data_ptr(),
                              total * sizeof(float), cudaMemcpyDeviceToDevice));
  }
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  silu_kernel<<<blocks, threads>>>(output->data_ptr(), total);
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaDeviceSynchronize());
}

// --------------------------------------------------
// multiply 算子实现（逐元素乘法）
// --------------------------------------------------
__global__ void multiply_kernel(const float *A, const float *B, float *out,
                                int total) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total) {
    out[idx] = A[idx] * B[idx];
  }
}

void multiply(Tensor<float> *output, const Tensor<float> *A,
              const Tensor<float> *B) {
  size_t total = A->numel();
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  multiply_kernel<<<blocks, threads>>>(A->data_ptr(), B->data_ptr(),
                                       output->data_ptr(), total);
  checkCudaError(cudaGetLastError());
  checkCudaError(cudaDeviceSynchronize());
}

// --------------------------------------------------
// 注意力计算：decode 版本 — 计算注意力分数
// --------------------------------------------------
__global__ void attention_scores_kernel(const float *Q, int n_q_h, int dqkv,
                                        const float *K, int cache_length,
                                        float *att_scores, int n_kv_h,
                                        bool is_3d_tensor) {
  int q = blockIdx.x;    // 查询头索引，每个 block 负责一个头
  int pos = threadIdx.x; // 缓存位置索引
  if (q < n_q_h && pos < cache_length) {
    // GQA映射: 完全匹配 CPU 版本
    int n_groups = n_q_h / n_kv_h; // 头分组数
    int kv_head = q / n_groups;    // 对应的 KV 头

    float dot = 0.0f;
    for (int i = 0; i < dqkv; i++) {
      float q_val;
      if (is_3d_tensor) {
        // 3D张量 [1, n_q_h, dqkv]，序列维度为 0
        q_val = Q[(0 * n_q_h + q) * dqkv + i];
      } else {
        // 2D张量 [n_q_h, dqkv]
        q_val = Q[q * dqkv + i];
      }
      // K 的布局：[cache_length, n_kv_h, dqkv]
      float k_val = K[(pos * n_kv_h + kv_head) * dqkv + i];
      dot += q_val * k_val;
    }
    att_scores[q * cache_length + pos] = dot / sqrtf((float)dqkv);
  }
}

// 计算注意力分数的函数
void compute_attention_scores(const Tensor<float> &Q, const Tensor<float> &K,
                              size_t n_q_h, size_t dqkv,
                              Tensor<float> &att_scores, size_t n_kv_h) {
  size_t cache_length = K.sizes()[0];

  // 检查 Q 的维度，支持 2D [n_heads, head_dim] 或 3D [1, n_heads, head_dim]
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

  // 每个查询头分配一个 block
  int blocks = n_q_h;
  int threads = std::min(static_cast<int>(cache_length), 1024);

  attention_scores_kernel<<<blocks, threads>>>(
      Q.data_ptr(), n_q_h, dqkv, K.data_ptr(), cache_length,
      att_scores.data_ptr(), n_kv_h, is_3d_q);

  checkCudaError(cudaGetLastError());
  checkCudaError(cudaDeviceSynchronize());
}
// --------------------------------------------------
// 注意力计算：decode 版本 — 计算注意力输出
// --------------------------------------------------
__global__ void att_output_kernel(const float *att_probs, int n_q_h,
                                  int cache_length, int dqkv, const float *V,
                                  float *att_output, int n_kv_h) {
  int q = blockIdx.x;
  int d = blockIdx.y;

  if (q < n_q_h && d < dqkv) {
    int n_groups = n_q_h / n_kv_h;
    int kv_head = q / n_groups;
    float sum = 0.0f;
    for (int pos = 0; pos < cache_length; ++pos) {
      float prob = att_probs[q * cache_length + pos];
      float val = V[(pos * n_kv_h + kv_head) * dqkv + d];
      sum += prob * val;
    }
    att_output[q * dqkv + d] = sum;
  }
}

void compute_att_output(const Tensor<float> &att_probs, const Tensor<float> &V,
                        size_t n_q_h, size_t dqkv, Tensor<float> &att_output,
                        size_t n_kv_h) {
  size_t cache_length = V.sizes()[0];

  // 验证维度是否匹配
  if (att_probs.sizes()[0] != n_q_h || att_probs.sizes()[1] != cache_length) {
    throw std::runtime_error(
        "attention probabilities tensor shape mismatch: expected [" +
        std::to_string(n_q_h) + ", " + std::to_string(cache_length) +
        "] but got [" + std::to_string(att_probs.sizes()[0]) + ", " +
        std::to_string(att_probs.sizes()[1]) + "]");
  }

  if (V.sizes()[0] != cache_length || V.sizes()[1] != n_kv_h ||
      V.sizes()[2] != dqkv) {
    throw std::runtime_error(
        "V tensor dimension mismatch: expected [" +
        std::to_string(cache_length) + ", " + std::to_string(n_kv_h) + ", " +
        std::to_string(dqkv) + "] but got [" + std::to_string(V.sizes()[0]) +
        ", " + std::to_string(V.sizes()[1]) + ", " +
        std::to_string(V.sizes()[2]) + "]");
  }

  if (att_output.sizes()[0] != n_q_h || att_output.sizes()[1] != dqkv) {
    throw std::runtime_error(
        "attention output tensor shape mismatch: expected [" +
        std::to_string(n_q_h) + ", " + std::to_string(dqkv) + "] but got [" +
        std::to_string(att_output.sizes()[0]) + ", " +
        std::to_string(att_output.sizes()[1]) + "]");
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
// 注意力计算：prefill 版本 — 计算注意力分数
// Q shape: [batch_size, n_q_h, dqkv], K shape: [cache_length, n_kv_h,
// dqkv], 输出 att_scores shape: [batch_size, n_q_h, cache_length]
__global__ void attention_scores_prefill_kernel(const float *Q, const float *K,
                                                float *att_scores, int n_q,
                                                int cache_length, int dqkv,
                                                int n_q_h, int n_kv_h) {
  int q = blockIdx.x;  // 扁平化后的查询索引：q in [0, n_q)
  int j = threadIdx.x; // 缓存位置索引，j in [0, cache_length)

  if (q < n_q && j < cache_length) {
    // 拆分出序列索引 s 和查询头索引 qh
    int s = q / n_q_h;
    int qh = q % n_q_h;

    // GQA 映射
    int n_groups = n_q_h / n_kv_h;
    int kv_head = qh / n_groups;

    float dot = 0.0f;
    // 为了调试，只打印第一个查询（s==0 且 qh==0）前两个特征的值
    for (int d = 0; d < dqkv; d++) {
      float q_val = Q[(s * n_q_h + qh) * dqkv + d];
      float k_val = K[(j * n_kv_h + kv_head) * dqkv + d];
      dot += q_val * k_val;
    }
    int out_idx = s * (n_q_h * cache_length) + qh * cache_length + j;
    att_scores[out_idx] = dot / sqrtf((float)dqkv);
  }
}

void compute_attention_scores_prefill(const Tensor<float> &Q,
                                      const Tensor<float> &K,
                                      Tensor<float> &att_scores, size_t dqkv) {
  // Q [seq_len, n_q_h, dqkv]
  // K [cache_length, n_kv_h, dqkv]
  // att_scores [seq_len, n_q_h, cache_length]

  // 获取维度信息
  size_t seq_len = Q.sizes()[0];
  size_t n_q_h = Q.sizes()[1];
  size_t cache_length = K.sizes()[0];
  size_t n_kv_h = K.sizes()[1];

  // 验证维度是否匹配
  if (Q.sizes()[2] != dqkv) {
    throw std::runtime_error("Q tensor dimension mismatch: expected " +
                             std::to_string(dqkv) + " but got " +
                             std::to_string(Q.sizes()[2]));
  }
  if (att_scores.sizes()[0] != seq_len || att_scores.sizes()[1] != n_q_h ||
      att_scores.sizes()[2] != cache_length) {
    throw std::runtime_error("attention scores tensor shape mismatch...");
  }
  if (K.sizes()[2] != dqkv) {
    throw std::runtime_error("K tensor dimension mismatch: expected " +
                             std::to_string(dqkv) + " but got " +
                             std::to_string(K.sizes()[2]));
  }

  if (att_scores.sizes()[0] != seq_len || att_scores.sizes()[1] != n_q_h ||
      att_scores.sizes()[2] != cache_length) {
    throw std::runtime_error(
        "attention scores tensor shape mismatch: expected [" +
        std::to_string(seq_len) + ", " + std::to_string(n_q_h) + ", " +
        std::to_string(cache_length) + "] but got [" +
        std::to_string(att_scores.sizes()[0]) + ", " +
        std::to_string(att_scores.sizes()[1]) + ", " +
        std::to_string(att_scores.sizes()[2]) + "]");
  }

  // 计算总查询数
  size_t total_q = seq_len * n_q_h;

  // 设置CUDA内核参数
  int threads =
      std::min(static_cast<int>(cache_length), 4096); // 最大线程数为1024
  int blocks = static_cast<int>(total_q);             // 每个查询一个block

  // 启动CUDA内核
  attention_scores_prefill_kernel<<<blocks, threads>>>(
      Q.data_ptr(), K.data_ptr(), att_scores.data_ptr(),
      static_cast<int>(total_q), static_cast<int>(cache_length),
      static_cast<int>(dqkv), static_cast<int>(n_q_h),
      static_cast<int>(n_kv_h));

  checkCudaError(cudaGetLastError());
  checkCudaError(cudaDeviceSynchronize());
}

// --------------------------------------------------
// 注意力计算：prefill 版本 — 计算注意力输出
// att_probs shape: [seq_len, n_q_h, cache_length], V shape:
// [cache_length, n_kv_h, dqkv], 输出 att_output shape: [seq_len, n_q_h,
// dqkv]
__global__ void att_output_prefill_kernel(const float *att_probs,
                                          const float *V, float *att_output,
                                          int n_q, int cache_length, int dqkv,
                                          int n_kv_h, int n_q_h) {
  int q = blockIdx.x;  // 查询索引
  int d = threadIdx.x; // 维度索引

  if (q < n_q && d < dqkv) {
    // 计算序列索引和查询头索引
    int s = q / n_q_h;  // 序列索引 (批次索引)
    int qh = q % n_q_h; // 查询头索引

    // GQA映射: 完全匹配CPU版本
    int n_groups = n_q_h / n_kv_h; // 头分组数
    int kv_head = qh / n_groups;   // 对应的KV头

    float sum = 0.0f;
    for (int j = 0; j < cache_length; j++) {
      // 使用与CPU相同的索引从3D张量访问注意力权重 - att_probs: [batch_size,
      // n_q_h, cache_length]
      int att_idx = s * (n_q_h * cache_length) + qh * cache_length + j;
      float prob = att_probs[att_idx];

      // V的布局：[cache_length, n_kv_h, dqkv]
      float val = V[(j * n_kv_h + kv_head) * dqkv + d];
      sum += prob * val;
    }

    // 写回结果到3D张量 - att_output: [batch_size, n_q_h, dqkv]
    int out_idx = (s * n_q_h + qh) * dqkv + d;
    att_output[out_idx] = sum;
  }
}

void compute_att_output_prefill(const Tensor<float> &att_probs,
                                const Tensor<float> &V,
                                Tensor<float> &att_output, size_t n_q_h,
                                size_t dqkv, size_t total_seq_len,
                                size_t n_kv_h) {
  // att_probs的形状应为[batch_size, n_q_h, cache_length]
  // V的形状应为[cache_length, n_kv_h, dqkv]
  // att_output的形状应为[batch_size, n_q_h, dqkv]

  // 获取维度信息
  size_t batch_size = att_probs.sizes()[0];
  size_t n_q_h_int = n_q_h;
  size_t cache_length = att_probs.sizes()[2];

  // 验证维度是否匹配
  if (att_probs.sizes()[1] != n_q_h_int) {
    throw std::runtime_error(
        "attention probabilities head dimension mismatch: expected " +
        std::to_string(n_q_h_int) + " but got " +
        std::to_string(att_probs.sizes()[1]));
  }

  if (V.sizes()[0] != cache_length || V.sizes()[1] != n_kv_h ||
      V.sizes()[2] != dqkv) {
    throw std::runtime_error(
        "V tensor dimension mismatch: expected [" +
        std::to_string(cache_length) + ", " + std::to_string(n_kv_h) + ", " +
        std::to_string(dqkv) + "] but got [" + std::to_string(V.sizes()[0]) +
        ", " + std::to_string(V.sizes()[1]) + ", " +
        std::to_string(V.sizes()[2]) + "]");
  }

  if (att_output.sizes()[0] != batch_size ||
      att_output.sizes()[1] != n_q_h_int || att_output.sizes()[2] != dqkv) {
    throw std::runtime_error(
        "attention output tensor shape mismatch: expected [" +
        std::to_string(batch_size) + ", " + std::to_string(n_q_h_int) + ", " +
        std::to_string(dqkv) + "] but got [" +
        std::to_string(att_output.sizes()[0]) + ", " +
        std::to_string(att_output.sizes()[1]) + ", " +
        std::to_string(att_output.sizes()[2]) + "]");
  }

  // 计算总查询数
  size_t total_q = batch_size * n_q_h_int;

  // 设置CUDA内核参数
  int threads = std::min(static_cast<int>(dqkv), 1024); // 最大线程数为1024
  int blocks = static_cast<int>(total_q); // 每个查询一个block

  // 启动CUDA内核
  att_output_prefill_kernel<<<blocks, threads>>>(
      att_probs.data_ptr(), V.data_ptr(), att_output.data_ptr(),
      static_cast<int>(total_q), static_cast<int>(cache_length),
      static_cast<int>(dqkv), static_cast<int>(n_kv_h),
      static_cast<int>(n_q_h_int));

  checkCudaError(cudaGetLastError());
  checkCudaError(cudaDeviceSynchronize());
}

} // namespace cuda_OP
