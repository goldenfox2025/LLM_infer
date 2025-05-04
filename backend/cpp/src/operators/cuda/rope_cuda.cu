#include <cmath>
#include <iostream>
#include <stdexcept>

#include "operators/cuda/rope_cuda.cuh"

namespace op {

// CUDA kernel for RoPE operation (optimized version)
template <typename T>
__global__ void rope_kernel_v1(T *tensor, size_t seq_len, size_t n_heads,
                               size_t head_dim, size_t offset, float theta) {
  // Each thread handles one dimension pair (i, i + head_dim/2) for a specific
  // (seq_idx, head_idx) Grid dimension should be (seq_len * n_heads * head_dim
  // / 2)
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total_rotations = seq_len * n_heads * (head_dim / 2);

  if (idx < total_rotations) {
    // Calculate position indices
    size_t dim_half = head_dim / 2;
    size_t rot_dim = idx % dim_half;
    size_t tmp = idx / dim_half;
    size_t head_idx = tmp % n_heads;
    size_t seq_idx = tmp / n_heads;

    // Calculate pointer to the start of the head
    T *head_ptr = tensor + seq_idx * n_heads * head_dim + head_idx * head_dim;

    // Calculate rotation parameters
    float freq = 1.0f / powf(theta, (2.0f * rot_dim) / head_dim);
    float val = (seq_idx + offset) * freq;
    float cos_val = cosf(val);
    float sin_val = sinf(val);

    // Apply rotation
    float x0 = static_cast<float>(head_ptr[rot_dim]);
    float x1 = static_cast<float>(head_ptr[rot_dim + dim_half]);
    head_ptr[rot_dim] = static_cast<T>(x0 * cos_val - x1 * sin_val);
    head_ptr[rot_dim + dim_half] = static_cast<T>(x0 * sin_val + x1 * cos_val);
  }
}

// Specialized kernel for BF16 data type
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
    // Calculate position indices
    size_t rot_dim_vec = idx % pairs_per_head_half;
    size_t tmp = idx / pairs_per_head_half;
    size_t head_idx = tmp % n_heads;
    size_t seq_idx = tmp / n_heads;

    // Calculate pointer to the start of the head
    __nv_bfloat16 *head_ptr =
        tensor + seq_idx * n_heads * head_dim + head_idx * head_dim;

    // Process two pairs at once (4 elements total)
    size_t rot_dim = rot_dim_vec * 2;

    // First pair
    float freq1 = 1.0f / powf(theta, (2.0f * rot_dim) / head_dim);
    float val1 = (seq_idx + offset) * freq1;
    float cos_val1 = cosf(val1);
    float sin_val1 = sinf(val1);

    float x0_1 = static_cast<float>(head_ptr[rot_dim]);
    float x1_1 = static_cast<float>(head_ptr[rot_dim + head_dim_half]);
    head_ptr[rot_dim] =
        static_cast<__nv_bfloat16>(x0_1 * cos_val1 - x1_1 * sin_val1);
    head_ptr[rot_dim + head_dim_half] =
        static_cast<__nv_bfloat16>(x0_1 * sin_val1 + x1_1 * cos_val1);

    // Second pair
    float freq2 = 1.0f / powf(theta, (2.0f * (rot_dim + 1)) / head_dim);
    float val2 = (seq_idx + offset) * freq2;
    float cos_val2 = cosf(val2);
    float sin_val2 = sinf(val2);

    float x0_2 = static_cast<float>(head_ptr[rot_dim + 1]);
    float x1_2 = static_cast<float>(head_ptr[rot_dim + 1 + head_dim_half]);
    head_ptr[rot_dim + 1] =
        static_cast<__nv_bfloat16>(x0_2 * cos_val2 - x1_2 * sin_val2);
    head_ptr[rot_dim + 1 + head_dim_half] =
        static_cast<__nv_bfloat16>(x0_2 * sin_val2 + x1_2 * cos_val2);
  }
}

// Generic kernel implementation
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

// Implementation of the RoPE CUDA operator - 使用二重指针以支持CUDA图优化
template <typename T>
void RopeCUDAOperator<T>::operator()(Tensor<T> **x_ptr, size_t *offset_ptr,
                                     float theta, cudaStream_t stream) {
  // 从二重指针获取实际值
  Tensor<T> *x = *x_ptr;
  size_t offset = *offset_ptr;

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
  } else
#endif
  {
    // Generic FP32/FP16 path
    size_t total_rotations =
        seq_len * n_heads * (head_dim / 2);  // Each thread handles one pair
    blocks = (total_rotations + threads - 1) / threads;
    kernel_ptr = (void *)rope_kernel_v1<T>;
  }

  // Kernel Launch
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

  // Error Checking
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "CUDA error after rope kernel launch: "
              << cudaGetErrorString(err) << std::endl;
    throw std::runtime_error("CUDA rope kernel launch failed");
  }
}

// Explicit template instantiations
template class RopeCUDAOperator<float>;
template class RopeCUDAOperator<__nv_bfloat16>;

}  // namespace op
