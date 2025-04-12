#ifndef CUDA_OP_CUH
#define CUDA_OP_CUH

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <float.h>

#include <limits>
#include <stdexcept>
#include <string>  // For std::to_string
#include <vector>

// Include CUB headers
#include <cub/cub.cuh>

#include "CudaMemoryPool.hpp"
#include "cudaOP.cuh"  // Assume common helpers might be here
#include "tensor.hpp"

#define MAX_TOPK \
  1024  // Maximum k handled by the fixed shared memory array in sampling kernel

// CUDA Error Check Macro
#define CUDA_CHECK(call)                                                \
  do {                                                                  \
    cudaError_t err = call;                                             \
    if (err != cudaSuccess) {                                           \
      fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(err));                                 \
      throw std::runtime_error(cudaGetErrorString(err));                \
    }                                                                   \
  } while (0)

namespace cuda_OP {

// Kernel 1 (Fused): Scale Logits and Initialize Indices (Multi-Block) - Same as
// before
template <typename T>
__global__ void scale_logits_and_init_indices_kernel(
    const T* __restrict__ logits, T* d_scaled_logits, int* d_indices,
    size_t vocab_size, float temperature) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;

  for (int i = idx; i < vocab_size; i += stride) {
    // Scale Logits
    float logit_f;
    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      logit_f = __bfloat162float(__ldg(&logits[i]));
    } else {
      logit_f = static_cast<float>(__ldg(&logits[i]));
    }
    float scaled_logit_f = logit_f / temperature;
    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      d_scaled_logits[i] = __float2bfloat16(scaled_logit_f);
    } else {
      d_scaled_logits[i] = static_cast<T>(scaled_logit_f);
    }
    // Initialize Indices
    d_indices[i] = i;
  }
}

// Kernel 2: Final Sampling from Top-K Results (Small Kernel)
// Templated on Block Size for CUB
template <typename T, int BLOCK_DIM_X>  // Template parameter for block size
__global__ void sample_from_sorted_topk_kernel(  // Renamed slightly for clarity
    const T* __restrict__ d_sorted_topk_logits,
    const int* __restrict__ d_sorted_topk_indices,
    size_t k,  // Note: k must be <= MAX_TOPK for this shared memory setup
    const float* __restrict__ d_max_val_ptr, curandState* states,
    uint32_t* d_sampled_index) {
  // Use CUB BlockReduce, templated on the known block size
  using BlockReduce = cub::BlockReduce<float, BLOCK_DIM_X>;

  // Allocate shared memory: CUB storage + space for k (up to MAX_TOPK) exp
  // values
  __shared__ union SharedStorage {
    typename BlockReduce::TempStorage reduce_storage;
    // Allocate enough space for MAX_TOPK floats AFTER the reduce storage
    // This requires knowing the size of reduce_storage.
    // A simpler, often sufficient approach if alignment works:
    struct Combined {
      typename BlockReduce::TempStorage reduce_storage;
      float exp_vals[MAX_TOPK];  // Fixed size array based on MAX_TOPK
    } combined;
  } shared_storage;

  int tid = threadIdx.x;

  // Shared variable to hold the max value read by thread 0
  __shared__ float max_val_shared;
  if (tid == 0) {
    max_val_shared = *d_max_val_ptr;
  }
  __syncthreads();  // Ensure all threads see max_val_shared

  // --- Parallel Calculation of exp(logit - max_val) for top k ---
  float thread_exp_sum = 0.0f;
  for (int i = tid; i < k; i += BLOCK_DIM_X) {  // Iterate up to actual k
    T scaled_logit_T = d_sorted_topk_logits[i];
    float scaled_logit_f;
    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      scaled_logit_f = __bfloat162float(scaled_logit_T);
    } else {
      scaled_logit_f = static_cast<float>(scaled_logit_T);
    }
    float exp_val = expf(scaled_logit_f - max_val_shared);

    // Store individual exp value in shared memory for later sampling scan
    // Ensure index 'i' is valid and within the allocated shared array size
    if (i < MAX_TOPK) {  // Check against allocated size
      shared_storage.combined.exp_vals[i] = exp_val;
    }
    // Accumulate sum locally for CUB reduction
    thread_exp_sum += exp_val;
  }
  __syncthreads();  // Ensure all exp_vals are written before reduction/sampling
                    // scan

  // --- Reduce the sum within the block using CUB ---
  float block_total_exp_sum =
      BlockReduce(shared_storage.combined.reduce_storage).Sum(thread_exp_sum);
  // block_total_exp_sum now holds the correct sum for all threads

  // --- Weighted sampling performed by thread 0 ---
  if (tid == 0) {
    float total_exp_sum = block_total_exp_sum;
    curandState localState = states[0];

    uint32_t selected_final_index = 0;

    if (total_exp_sum <= 1e-9f || k == 0) {
      if (k > 0) {
        selected_final_index = static_cast<uint32_t>(d_sorted_topk_indices[0]);
      } else {
        selected_final_index = 0;
      }
    } else {
      float r = curand_uniform(&localState) * total_exp_sum;
      float cumulative = 0.0f;

      // Linear scan through top-k exp values stored in shared memory
      selected_final_index = static_cast<uint32_t>(d_sorted_topk_indices[0]);
      // Access the exp_vals array within the shared structure
      float* s_exp_vals = shared_storage.combined.exp_vals;
      for (int i = 0; i < k; ++i) {
        // Ensure i does not exceed MAX_TOPK (boundary check already done by k)
        cumulative += s_exp_vals[i];  // Read from shared memory
        if (cumulative >= r) {
          selected_final_index =
              static_cast<uint32_t>(d_sorted_topk_indices[i]);
          break;
        }
      }
    }
    *d_sampled_index = selected_final_index;
    states[0] = localState;
  }
}

// Functor for CUB TransformIterator - Same as before
template <typename Tin>
struct ConvertToFloatFunctor {
  __device__ __forceinline__ float operator()(const Tin& x) const {
    if constexpr (std::is_same_v<Tin, __nv_bfloat16>) {
      return __bfloat162float(x);
    } else {
      return static_cast<float>(x);
    }
  }
};

template <typename T>
uint32_t* sample(Tensor<T>&& logits, float temperature,
                float top_p,  // top_p still unused
                size_t top_k, curandState* d_states) {
  if (logits.device() != Device::CUDA) {
    throw std::runtime_error("Input tensor must be on CUDA device");
  }
  // Allow top_k == 0 conceptually, but kernel needs k >= 1 if sampling is done
  // Let's enforce k>=1 here, or handle k=0 case explicitly if needed (e.g.
  // return default index)
  if (top_k == 0) {
    throw std::runtime_error("top_k must be at least 1 for sampling");
  }

  const auto& shape = logits.sizes();
  if (shape.size() != 2 || shape[0] == 0 || shape[1] == 0) {
    throw std::runtime_error(
        "Input tensor must be 2D with non-zero dimensions [seq_len, "
        "vocab_size]");
  }

  const size_t seq_len = shape[0];
  const size_t vocab_size = shape[1];
  if (top_k > vocab_size) {
    top_k = vocab_size;  // Clamp k to vocab size
  }
  // Clamp k based on shared memory limit for sampling kernel
  if (top_k > MAX_TOPK) {
    // Option 1: Clamp k (may affect desired behavior if user requested large k)
    // std::cerr << "Warning: Requested top_k (" << top_k
    //           << ") exceeds kernel limit MAX_TOPK (" << MAX_TOPK
    //           << "). Clamping k." << std::endl;
    // top_k = MAX_TOPK;
    // Option 2: Throw error (Safer - user needs to adjust k or kernel)
    throw std::runtime_error("Requested top_k (" + std::to_string(top_k) +
                             ") exceeds kernel's MAX_TOPK limit (" +
                             std::to_string(MAX_TOPK) +
                             ") for shared memory allocation.");
  }

  const T* d_logits_ptr = logits.data_ptr() + (seq_len - 1) * vocab_size;

  // --- Memory Allocation ---
  auto& pool = GlobalCudaMemoryPool::instance();
  T* d_scaled_logits = static_cast<T*>(pool.allocate(vocab_size * sizeof(T)));
  float* d_max_val = static_cast<float*>(pool.allocate(sizeof(float)));
  int* d_indices = static_cast<int*>(pool.allocate(vocab_size * sizeof(int)));
  T* d_sorted_logits = static_cast<T*>(pool.allocate(vocab_size * sizeof(T)));
  int* d_sorted_indices =
      static_cast<int*>(pool.allocate(vocab_size * sizeof(int)));
  uint32_t* d_sampled_index =
      static_cast<uint32_t*>(pool.allocate(sizeof(uint32_t)));

  void* d_reduce_temp_storage = nullptr;
  size_t reduce_temp_storage_bytes = 0;
  void* d_sort_temp_storage = nullptr;
  size_t sort_temp_storage_bytes = 0;

  cudaStream_t stream = nullptr;

  // --- Step 1 (Fused): Scale Logits and Initialize Indices ---
  const int scale_init_block_size = 256;
  const int scale_init_grid_size =
      (vocab_size + scale_init_block_size - 1) / scale_init_block_size;
  scale_logits_and_init_indices_kernel<T>
      <<<scale_init_grid_size, scale_init_block_size, 0, stream>>>(
          d_logits_ptr, d_scaled_logits, d_indices, vocab_size, temperature);
  CUDA_CHECK(cudaGetLastError());

  // --- Step 2: Find Max Scaled Logit (CUB Device Reduce) ---
  cub::TransformInputIterator<float, ConvertToFloatFunctor<T>, const T*> itr(
      d_scaled_logits, ConvertToFloatFunctor<T>());
  CUDA_CHECK(cub::DeviceReduce::Max(d_reduce_temp_storage,
                                    reduce_temp_storage_bytes, itr, d_max_val,
                                    vocab_size, stream));
  d_reduce_temp_storage = pool.allocate(reduce_temp_storage_bytes);
  CUDA_CHECK(cub::DeviceReduce::Max(d_reduce_temp_storage,
                                    reduce_temp_storage_bytes, itr, d_max_val,
                                    vocab_size, stream));
  CUDA_CHECK(cudaGetLastError());

  // --- Step 3: Sort (Scaled Logit, Index) Pairs Descending ---
  CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(
      d_sort_temp_storage, sort_temp_storage_bytes, d_scaled_logits,
      d_sorted_logits, d_indices, d_sorted_indices, vocab_size, 0,
      sizeof(T) * 8, stream));
  d_sort_temp_storage = pool.allocate(sort_temp_storage_bytes);
  CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(
      d_sort_temp_storage, sort_temp_storage_bytes, d_scaled_logits,
      d_sorted_logits, d_indices, d_sorted_indices, vocab_size, 0,
      sizeof(T) * 8, stream));
  CUDA_CHECK(cudaGetLastError());

  // --- Step 4: Final Weighted Sampling ---
  // Choose block size for the sampling kernel
  const int sample_block_size = 128;  // Example size (must match template arg)

  // Calculate required shared memory
  // Need size of CUB reduce storage for sample_block_size + size of exp_vals
  // array Getting reduce storage size precisely requires template
  // specialization or careful estimation. Let's estimate based on common block
  // sizes or use a helper if CUB provides one. For now, use a reasonable upper
  // bound estimate or sizeof on a concrete type. sizeof(cub::BlockReduce<float,
  // 128>::TempStorage) - check documentation or experiment Example estimation:
  size_t reduce_storage_size_est = sizeof(
      cub::BlockReduce<float, sample_block_size>::TempStorage);  // If known at
                                                                 // compile time
  size_t exp_values_size =
      MAX_TOPK * sizeof(float);  // Based on the kernel's array
  size_t sample_shared_mem =
      reduce_storage_size_est + exp_values_size;  // Total needed

  // Check against device limits
  int max_shared_mem_per_block = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&max_shared_mem_per_block,
                                    cudaDevAttrMaxSharedMemoryPerBlock, 0));

  if (sample_shared_mem > max_shared_mem_per_block) {
    throw std::runtime_error("Calculated shared memory required (" +
                             std::to_string(sample_shared_mem) +
                             ") exceeds device limit (" +
                             std::to_string(max_shared_mem_per_block) +
                             "). Check MAX_TOPK or reduce sample_block_size.");
  }

  // Launch the kernel, passing block size as template argument
  sample_from_sorted_topk_kernel<T, sample_block_size>
      <<<1, sample_block_size, sample_shared_mem, stream>>>(
          d_sorted_logits, d_sorted_indices,
          top_k,  // Pass the actual k requested (already clamped <= MAX_TOPK)
          d_max_val, d_states, d_sampled_index);
  CUDA_CHECK(cudaGetLastError());

  // --- Copy result back ---
  // uint32_t h_result = 0;
  // CUDA_CHECK(cudaMemcpy(&h_result, d_sampled_index, sizeof(uint32_t),
  //                       cudaMemcpyDeviceToHost));

  // --- Free Memory ---
  pool.free(d_scaled_logits);
  pool.free(d_max_val);
  pool.free(d_indices);
  pool.free(d_sorted_logits);
  pool.free(d_sorted_indices);
  // pool.free(d_sampled_index);
  pool.free(d_reduce_temp_storage);
  pool.free(d_sort_temp_storage);

  return d_sampled_index;
}

// Template instantiations
template uint32_t* sample<float>(Tensor<float>&&, float, float, size_t,
                                curandState*);
template uint32_t* sample<__nv_bfloat16>(Tensor<__nv_bfloat16>&&, float, float,
                                        size_t, curandState*);

}  // namespace cuda_OP

#endif  // CUDA_OP_CUH