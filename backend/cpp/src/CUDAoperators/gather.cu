#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>  // printf
#include <iostream>
#include <stdexcept>
#include <vector>

#include "cudaOP.cuh"
#include "tensor.hpp"
namespace cuda_OP {

// 模板化的 CUDA kernel，output[i, :] = embedding_table[input[i], :]
template <typename T>
__global__ void gather_kernel_v1(const uint32_t* input,
                                 const T* embedding_table, T* output,
                                 int seq_len, int embed_dim, int vocab_size) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int total = seq_len * embed_dim;
  for (int idx = tid; idx < total; idx += stride) {
    int row = idx / embed_dim;
    int col = idx % embed_dim;
    // 越界检查（理论上 grid-stride loop
    // 已经保证了，不需要再次判断，但增加健壮性）
    if (row >= seq_len || col >= embed_dim) {
      continue;
    }
    uint32_t token_id = input[row];
    if (token_id >= vocab_size) {
      continue;
    }
    int emb_index = token_id * embed_dim + col;
    // 再次检查防止溢出
    if (emb_index >= vocab_size * embed_dim) {
      continue;
    }
    T value = embedding_table[emb_index];
    output[idx] = value;
  }
}

// 模板化的 host 端 gather 函数
template <typename T>
void gather(Tensor<T>* output, const Tensor<uint32_t>* input,
            const Tensor<T>* embedding_table) {
  int seq_len = static_cast<int>(input->numel());
  int embed_dim = static_cast<int>(output->sizes()[1]);
  int vocab_size = static_cast<int>(embedding_table->sizes()[0]);

  // 验证设备一致性
  if (input->device() != Device::CUDA) {
    throw std::runtime_error("Input must be on CUDA device");
  }
  if (output->device() != Device::CUDA) {
    throw std::runtime_error("Output must be on CUDA device");
  }
  if (embedding_table->device() != Device::CUDA) {
    throw std::runtime_error("Embedding table must be on CUDA device");
  }

  // 验证输入数据有效性
  if (seq_len <= 0 || embed_dim <= 0 || vocab_size <= 0) {
    throw std::runtime_error("Invalid dimensions in gather");
  }

  int threadsPerBlock = 256;
  int total = seq_len * embed_dim;
  int blocks = (total + threadsPerBlock - 1) / threadsPerBlock;

  // 启动 CUDA kernel
  gather_kernel_v1<T><<<blocks, threadsPerBlock>>>(
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
template void gather<nvbf16>(Tensor<nvbf16>*, const Tensor<uint32_t>*,
                             const Tensor<nvbf16>*);
template void gather<float>(Tensor<float>*, const Tensor<uint32_t>*,
                            const Tensor<float>*);
}  // namespace cuda_OP
