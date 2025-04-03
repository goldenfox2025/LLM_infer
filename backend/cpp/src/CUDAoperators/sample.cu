#ifndef CUDA_OP_CUH
#define CUDA_OP_CUH

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <float.h>
#include <limits>
#include <stdexcept>
#include <vector>

// Include CUB headers for reduction and sorting
#include <cub/cub.cuh>

#include "CudaMemoryPool.hpp"
#include "cudaOP.cuh" // Assume common helpers might be here
#include "tensor.hpp"

#define MAX_TOPK 1024 // Keep this for potential clamp

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

// Kernel 1: Scale Logits (Multi-Block) - Same as before
template <typename T>
__global__ void scale_logits_kernel(const T* __restrict__ logits,
                                     T* d_scaled_logits,
                                     size_t vocab_size,
                                     float temperature)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < vocab_size; i += stride) {
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
    }
}

// Kernel 2: Initialize Indices (Multi-Block)
__global__ void init_indices_kernel(int* d_indices, size_t vocab_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < vocab_size; i += stride) {
        d_indices[i] = i;
    }
}


// Kernel 3: Final Sampling from Top-K Results (Small Kernel)
// Takes the *sorted* top K scaled logits and their *original* indices
template <typename T>
__global__ void sample_from_sorted_topk_kernel(
    const T* __restrict__ d_sorted_topk_logits, // Top K scaled+sorted logits values
    const int* __restrict__ d_sorted_topk_indices, // Original indices of top K (now sorted by logit value)
    size_t k,
    const float* __restrict__ d_max_val_ptr, // Pointer to the single max value (for stability)
    curandState* states,                     // curand state (input/output)
    uint32_t* d_sampled_index                // Output index
) {
    // Kernel runs with <<<1, block_size>>> where block_size <= 1024
    // Shared memory to store intermediate exp() results for the top K items
    extern __shared__ float s_exp_vals[]; // Size k*sizeof(float) needed
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    // Thread 0 reads the global max value (previously computed)
    __shared__ float max_val_shared;
    if (tid == 0) {
        max_val_shared = *d_max_val_ptr;
    }
    __syncthreads(); // Ensure all threads see max_val_shared

    // --- Parallel Calculation of exp(logit - max_val) for top k ---
    // Each thread calculates some exp values up to k
    float thread_partial_sum = 0.0f;
    for (int i = tid; i < k; i += block_size) {
        T scaled_logit_T = d_sorted_topk_logits[i];
        float scaled_logit_f;
         if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            scaled_logit_f = __bfloat162float(scaled_logit_T);
         } else {
            scaled_logit_f = static_cast<float>(scaled_logit_T);
         }
        float exp_val = expf(scaled_logit_f - max_val_shared);
        s_exp_vals[i] = exp_val; // Store individual exp value in shared memory
        thread_partial_sum += exp_val; // Accumulate sum locally for reduction
    }
    __syncthreads(); // Ensure all exp values are written to s_exp_vals

    // --- Reduce the sum within the block ---
    // Using warp shuffle reduction + shared memory for cross-warp reduction
    float block_sum = 0;
    if (block_size >= warpSize) // Check if shuffles are needed/useful
    {
         float warp_sum = thread_partial_sum;
         // Reduce within warps
         for(int offset = warpSize/2; offset > 0; offset /= 2) {
            warp_sum += __shfl_down_sync(0xFFFFFFFF, warp_sum, offset);
         }
          // Warp leaders write to shared memory (use first k slots reused)
         __shared__ float s_warp_sums[32]; // Max 32 warps for block size <= 1024
         if((tid % warpSize) == 0) {
              s_warp_sums[tid / warpSize] = warp_sum;
         }
          __syncthreads();

         // Reduce across warps by thread 0
          if(tid == 0) {
              int num_warps = block_size / warpSize;
              block_sum = 0.0f;
              for(int i = 0; i < num_warps; ++i) {
                  block_sum += s_warp_sums[i];
              }
          }
    } else { // Small block size, thread 0 can sum directly if needed
         if (tid == 0) {
            block_sum = 0.0f;
            for(int i=0; i < block_size; ++i) {
                // Need a way to get other threads' partial sums
                // Alternative: Just sum from s_exp_vals directly
                // This avoids complex reduction code if k is small
                for (int j=0; j<k; ++j) { // Sum all k exp values
                    block_sum += s_exp_vals[j];
                }
            }
         }
    }
    // Simpler alternative for reduction if k <= 1024 and block_size is reasonable:
    // Have thread 0 sum the k values directly from shared memory after they are calculated.
    __shared__ float final_sum_shared;
    if (tid == 0) {
        float sum_k = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum_k += s_exp_vals[i];
        }
        final_sum_shared = sum_k; // Store final sum for sampling
    }
    __syncthreads(); // Ensure sum is calculated and visible

    // --- Weighted sampling performed by thread 0 ---
    if (tid == 0) {
        float total_exp_sum = final_sum_shared; // Get the final sum
        curandState localState = states[0]; // Load state

        uint32_t selected_final_index = 0; // Default

        if (total_exp_sum <= 1e-9f || k == 0) { // Use a small epsilon for float comparison
            // Handle edge case: No valid candidates or zero probability sum.
            // Default to the highest probability one (which is index 0 of the *sorted* arrays) if k>0.
             if (k > 0) {
                selected_final_index = static_cast<uint32_t>(d_sorted_topk_indices[0]);
             } else {
                selected_final_index = 0; // Or error indicator
             }
        } else {
            // Generate random number scaled by the sum of top-k exponentials
            float r = curand_uniform(&localState) * total_exp_sum;
            float cumulative = 0.0f;

            // Linear scan through top-k *shared memory* exp values
            selected_final_index = static_cast<uint32_t>(d_sorted_topk_indices[0]); // Default to first
            for (int i = 0; i < k; ++i) {
                cumulative += s_exp_vals[i]; // Use the stored exp value from shared memory
                if (cumulative >= r) {
                    selected_final_index = static_cast<uint32_t>(d_sorted_topk_indices[i]);
                    break;
                }
            }
        }

        // Write the final selected index to global memory
        *d_sampled_index = selected_final_index;

        // Store the updated random state back
        states[0] = localState;
    }
}


// Functor for CUB TransformIterator (Corrected)
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
uint32_t sample(Tensor<T>&& logits, float temperature, float top_p, // top_p still unused
                size_t top_k, curandState* d_states) {
    if (logits.device() != Device::CUDA) {
        throw std::runtime_error("Input tensor must be on CUDA device");
    }
    if (top_k == 0) {
         throw std::runtime_error("top_k must be at least 1");
    }


    const auto& shape = logits.sizes();
    if (shape.size() != 2 || shape[0] == 0 || shape[1] == 0) {
         throw std::runtime_error("Input tensor must be 2D with non-zero dimensions [seq_len, vocab_size]");
    }

    const size_t seq_len = shape[0];
    const size_t vocab_size = shape[1];
    // Clamp top_k to vocab_size if it's larger
    if (top_k > vocab_size) {
        top_k = vocab_size;
    }

    const T* d_logits_ptr = logits.data_ptr() + (seq_len - 1) * vocab_size; // Pointer to last row

    // --- Memory Allocation ---
    auto& pool = GlobalCudaMemoryPool::instance();
    T* d_scaled_logits = static_cast<T*>(pool.allocate(vocab_size * sizeof(T)));
    float* d_max_val = static_cast<float*>(pool.allocate(sizeof(float))); // For max logit result
    int* d_indices = static_cast<int*>(pool.allocate(vocab_size * sizeof(int))); // Indices 0..N-1
    // Buffers for sorted results (need full size for sort, even if only using top k later)
    T* d_sorted_logits = static_cast<T*>(pool.allocate(vocab_size * sizeof(T)));
    int* d_sorted_indices = static_cast<int*>(pool.allocate(vocab_size * sizeof(int)));
    uint32_t* d_sampled_index = static_cast<uint32_t*>(pool.allocate(sizeof(uint32_t)));

    void* d_reduce_temp_storage = nullptr;
    size_t reduce_temp_storage_bytes = 0;
    void* d_sort_temp_storage = nullptr;
    size_t sort_temp_storage_bytes = 0;

    cudaStream_t stream = 0; // Use default stream for simplicity

    // --- Step 1: Scale Logits (Multi-Block Kernel) ---
    const int scale_block_size = 256;
    const int scale_grid_size = (vocab_size + scale_block_size - 1) / scale_block_size;
    scale_logits_kernel<T><<<scale_grid_size, scale_block_size, 0, stream>>>(
        d_logits_ptr, d_scaled_logits, vocab_size, temperature);
    CUDA_CHECK(cudaGetLastError());

    // --- Step 2: Find Max Scaled Logit (CUB Device Reduce) ---
    // Use the corrected TransformInputIterator with the functor
    cub::TransformInputIterator<float, ConvertToFloatFunctor<T>, const T*> itr(d_scaled_logits, ConvertToFloatFunctor<T>());

    CUDA_CHECK(cub::DeviceReduce::Max(d_reduce_temp_storage, reduce_temp_storage_bytes,
                                      itr, d_max_val, vocab_size, stream)); // Pass iterator
    d_reduce_temp_storage = pool.allocate(reduce_temp_storage_bytes);
    CUDA_CHECK(cub::DeviceReduce::Max(d_reduce_temp_storage, reduce_temp_storage_bytes,
                                      itr, d_max_val, vocab_size, stream)); // Pass iterator
    CUDA_CHECK(cudaGetLastError());

    // --- Step 3: Initialize Indices 0..N-1 ---
    const int init_block_size = 256;
    const int init_grid_size = (vocab_size + init_block_size - 1) / init_block_size;
    init_indices_kernel<<<init_grid_size, init_block_size, 0, stream>>>(d_indices, vocab_size);
    CUDA_CHECK(cudaGetLastError());

    // --- Step 4: Sort (Scaled Logit, Index) Pairs Descending (CUB Device Radix Sort) ---
    // Correct CUB API: SortPairsDescending
    CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(
        d_sort_temp_storage, sort_temp_storage_bytes,
        d_scaled_logits, d_sorted_logits,  // Input keys -> Output keys
        d_indices, d_sorted_indices,       // Input values -> Output values
        vocab_size,
        0, // begin_bit
        sizeof(T) * 8, // end_bit (adjust if T is not float/int based) -> Use appropriate bits for T
                       // For float/bfloat16, sorting often uses integer representations, so full bits work.
        stream));

    d_sort_temp_storage = pool.allocate(sort_temp_storage_bytes);

    CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(
        d_sort_temp_storage, sort_temp_storage_bytes,
        d_scaled_logits, d_sorted_logits,  // Input keys -> Output keys
        d_indices, d_sorted_indices,       // Input values -> Output values
        vocab_size,
        0, sizeof(T) * 8, // Check bits calculation - might need adjustment for bfloat16? CUB usually handles float/int keys well.
        stream));
    CUDA_CHECK(cudaGetLastError());

    // --- Step 5: Final Weighted Sampling (Small Kernel on Top K Results) ---
    // We now pass the *start* of the sorted buffers to the kernel.
    // It will only read the first 'k' elements it needs.
    const int sample_block_size = 128; // Can tune this (e.g., 64, 128, 256)
    // Shared memory needed: k * sizeof(float)
    size_t sample_shared_mem = top_k * sizeof(float);
    // Add check/clamp for excessive shared memory request if needed
     if (sample_shared_mem > 48000) { // e.g., limit for older GPUs
         // Handle error or adjust strategy (e.g., serial sampling in thread 0)
         // For now, assume k * sizeof(float) is acceptable
         // Alternative: Reduce sample_block_size, but sampling only needs k values.
         sample_shared_mem = 48000; // Clamp if necessary, though kernel logic relies on k floats
         // Better: throw error if k is too large for shared mem approach
         if (top_k > (48000 / sizeof(float))) {
              throw std::runtime_error("top_k value too large for shared memory sampling kernel");
         }
     }


    sample_from_sorted_topk_kernel<T><<<1, sample_block_size, sample_shared_mem, stream>>>(
        d_sorted_logits,    // Pass pointer to start of sorted logits buffer
        d_sorted_indices,   // Pass pointer to start of sorted indices buffer
        top_k,              // Kernel only processes first k elements
        d_max_val,
        d_states,
        d_sampled_index);
    CUDA_CHECK(cudaGetLastError());

    // --- Copy result back ---
    uint32_t h_result = 0;
    // Ensure kernel is complete before copy if using non-default streams
    // CUDA_CHECK(cudaStreamSynchronize(stream)); // Needed if stream != 0
    CUDA_CHECK(cudaMemcpy(&h_result, d_sampled_index, sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // --- Free Memory ---
    pool.free(d_scaled_logits);
    pool.free(d_max_val);
    pool.free(d_indices);
    pool.free(d_sorted_logits);
    pool.free(d_sorted_indices);
    pool.free(d_sampled_index);
    pool.free(d_reduce_temp_storage);
    pool.free(d_sort_temp_storage);

    return h_result;
}

// Template instantiations
template uint32_t sample<float>(Tensor<float>&&, float, float, size_t,
                                curandState*);
template uint32_t sample<__nv_bfloat16>(Tensor<__nv_bfloat16>&&, float, float,
                                        size_t, curandState*);

}  // namespace cuda_OP

#endif  // CUDA_OP_CUHs