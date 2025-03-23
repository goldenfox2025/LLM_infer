
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <math.h>

#include <cstdio>  // //printf
#include <iostream>
#include <stdexcept>
#include <vector>

#include "cudaOP.cuh"

// --------------------------------------------------
// gather 算子实现
// --------------------------------------------------
// kernel：output[i, :] = embedding_table[input[i], :]
// CUDA kernel 使用二维线程块与网格

// embedding_table: [vocab_size, embed_dim]
// input: [seq_len] (里面是 token_id)
// output: [seq_len, embed_dim]
namespace cuda_OP {
__global__ void gather_kernel_v1(const uint32_t *input,
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

  gather_kernel_v1<<<blocks, threadsPerBlock>>>(
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
}  // namespace cuda_OP