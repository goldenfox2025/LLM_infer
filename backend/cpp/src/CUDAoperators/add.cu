#include <cuda_bf16.h>  // Required for nv_bfloat16, nv_bfloat162, __hadd2
#include <cuda_runtime.h>
#include <math.h>

#include <algorithm>  // For std::min
#include <cstdio>
#include <iostream>
#include <stdexcept>    // For std::runtime_error
#include <type_traits>  // For std::is_same_v
#include <vector>

// Assume cudaOP.cuh contains Tensor definition, checkCudaError, etc.
#include "cudaOP.cuh"

namespace cuda_OP {

// --------------------------------------------------
// v1: Grid-Stride Loop Kernel (No __ldg - Benchmark vs original!)
// --------------------------------------------------
template <typename T>
__global__ void add_kernel_v1(const T *__restrict__ A, const T *__restrict__ B,
                              T *__restrict__ out, size_t total) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;

  for (size_t i = idx; i < total; i += stride) {
    // Standard load - relies on L1/L2 cache, often good for coalesced access
    out[i] = A[i] + B[i];
  }
}

// --------------------------------------------------
// v2: Vectorized (x2) Grid-Stride Loop Kernels
// --------------------------------------------------

// v2 Kernel for float (using float2)
__global__ void add_kernel_float_v2(
    const float2 *__restrict__ A, const float2 *__restrict__ B,
    float2 *__restrict__ out,
    size_t total_vec2)  // total number of float2 elements
{
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;

  for (size_t i = idx; i < total_vec2; i += stride) {
    float2 a_val = A[i];
    float2 b_val = B[i];
    float2 result;
    result.x = a_val.x + b_val.x;
    result.y = a_val.y + b_val.y;
    out[i] = result;
  }
}

// v2 Kernel for bfloat16 (using nv_bfloat162)
__global__ void add_kernel_bf16_v2(
    const nv_bfloat162 *__restrict__ A, const nv_bfloat162 *__restrict__ B,
    nv_bfloat162 *__restrict__ out,
    size_t total_vec2)  // total number of nv_bfloat162 elements
{
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;

  for (size_t i = idx; i < total_vec2; i += stride) {
#if __CUDA_ARCH__ >= 800           // Use intrinsic on Ampere (SM 8.0) and later
    out[i] = __hadd2(A[i], B[i]);  // Hardware accelerated 2xBF16 add
#else
    // Fallback for older architectures (less efficient)
    // Note: Direct addition of nv_bfloat16 might involve implicit casts
    // or require explicit handling depending on compiler settings.
    // __hadd2 is strongly preferred.
    nv_bfloat162 a_val = A[i];
    nv_bfloat162 b_val = B[i];
    nv_bfloat162 result;
    // Perform addition (may involve intermediate float conversion implicitly)
    result.x = a_val.x + b_val.x;
    result.y = a_val.y + b_val.y;
    out[i] = result;
#endif
  }
}

// --------------------------------------------------
// Host function using the v2 strategy (vectorization + grid-stride)
// Falls back to v1 (grid-stride scalar) if vectorization isn't suitable.
// Includes stream parameter for asynchronous execution.
// --------------------------------------------------
template <typename T>
void add(Tensor<T> *output, Tensor<T> *A, Tensor<T> *B, cudaStream_t stream) {
  // --- Basic Checks ---
  if (A->numel() != B->numel() || A->numel() != output->numel()) {
    throw std::runtime_error("Tensor shapes must match for addition");
  }
  if (A->device() != Device::CUDA || B->device() != Device::CUDA ||
      output->device() != Device::CUDA) {
    throw std::runtime_error("All tensors must be on CUDA device for addition");
  }

  size_t total = A->numel();
  if (total == 0) {
    return;  // Nothing to do
  }

  // --- Kernel Launch Configuration ---
  int threads = 256;  // Good starting point, tune this (128, 512?)

  // --- Determine Optimal Kernel ---
  bool can_vectorize = false;
  size_t vec_alignment = 0;

  if constexpr (std::is_same_v<T, float>) {
    vec_alignment = alignof(float2);  // Usually 8
    can_vectorize = true;
  } else if constexpr (std::is_same_v<T, nvbf16>) {
    vec_alignment = alignof(nv_bfloat162);  // Usually 4
    can_vectorize = true;
  }

  // Check conditions for vectorization:
  // 1. Type supported (float or nvbf16 here)
  // 2. Total elements is even
  // 3. Data pointers are aligned to the vector type's requirement
  bool use_vectorized_kernel =
      can_vectorize && (total % 2 == 0) &&
      (reinterpret_cast<uintptr_t>(A->data_ptr()) % vec_alignment == 0) &&
      (reinterpret_cast<uintptr_t>(B->data_ptr()) % vec_alignment == 0) &&
      (reinterpret_cast<uintptr_t>(output->data_ptr()) % vec_alignment == 0);

  // --- Launch Kernel ---
  if (use_vectorized_kernel) {
    size_t total_vec2 = total / 2;

    // Calculate blocks using a heuristic for grid-stride
    // Aim for enough blocks to saturate the GPU. A common heuristic is
    // a multiple of the number of SMs.
    int device;
    checkCudaError(cudaGetDevice(&device));
    int numSMs;
    checkCudaError(cudaDeviceGetAttribute(
        &numSMs, cudaDevAttrMultiProcessorCount, device));
    // e.g., aim for 32 waves of blocks across the GPU, minimum 1 block
    int blocks = std::max(
        1, std::min((int)((total_vec2 + threads - 1) / threads), numSMs * 32));

    if constexpr (std::is_same_v<T, float>) {
      add_kernel_float_v2<<<blocks, threads, 0, stream>>>(
          reinterpret_cast<const float2 *>(A->data_ptr()),
          reinterpret_cast<const float2 *>(B->data_ptr()),
          reinterpret_cast<float2 *>(output->data_ptr()), total_vec2);
    } else if constexpr (std::is_same_v<T, nvbf16>) {
      add_kernel_bf16_v2<<<blocks, threads, 0, stream>>>(
          reinterpret_cast<const nv_bfloat162 *>(A->data_ptr()),
          reinterpret_cast<const nv_bfloat162 *>(B->data_ptr()),
          reinterpret_cast<nv_bfloat162 *>(output->data_ptr()), total_vec2);
    }
  } else {
    // Fallback to scalar grid-stride kernel (v1)
    int device;
    checkCudaError(cudaGetDevice(&device));
    int numSMs;
    checkCudaError(cudaDeviceGetAttribute(
        &numSMs, cudaDevAttrMultiProcessorCount, device));
    int blocks = std::max(
        1, std::min((int)((total + threads - 1) / threads), numSMs * 32));

    add_kernel_v1<T><<<blocks, threads, 0, stream>>>(
        A->data_ptr(), B->data_ptr(), output->data_ptr(), total);
  }

  // Check for asynchronous errors (from kernel launch)
  checkCudaError(cudaGetLastError());

  // NOTE: No cudaDeviceSynchronize() here! Synchronization should be managed
  // by the caller using the stream if needed (e.g.,
  // cudaStreamSynchronize(stream)).
}

// Explicit instantiations for the host function add_v2
template void add<float>(Tensor<float> *, Tensor<float> *, Tensor<float> *,
                         cudaStream_t);
template void add<nvbf16>(Tensor<nvbf16> *, Tensor<nvbf16> *, Tensor<nvbf16> *,
                          cudaStream_t);

}  // namespace cuda_OP