#include <cmath>
#include <iostream>
#include <stdexcept>

#include "cudaOP.cuh"
#include "ptx_common.h"

namespace cuda_OP {

// A simplified and optimized kernel for fused RoPE and KV Cache writing.
// It processes Q and K with RoPE, then writes K and V into their respective layer caches.
// It's designed to be called per layer, receiving pointers to that layer's cache.
template <typename T, int actual_pairs_per_thread = 2>
__global__ void rope_kv_fused_kernel(
    T* q_inout,           // Q tensor data, modified in-place
    const T* k_input,     // K tensor data
    const T* v_input,     // V tensor data
    T* k_cache_layer_ptr, // Pointer to the start of the K cache for the current layer
    T* v_cache_layer_ptr, // Pointer to the start of the V cache for the current layer
    size_t seq_len,
    size_t n_heads,
    size_t n_kv_heads,
    size_t head_dim,
    const size_t* d_rope_offset, // Past sequence length (offset in cache)
    const float* sin_cos_cache,  // Precomputed sin/cos values
    size_t sin_cos_stride,       // Stride for sin_cos_cache
    size_t kv_cache_seq_stride,  // Stride for one sequence element in the cache
    size_t q_stride_s,           // Sequence stride for Q tensor
    size_t k_stride_s,           // Sequence stride for K tensor
    size_t v_stride_s            // Sequence stride for V tensor
) {
    // Use shared memory for efficient V-cache writing
    extern __shared__ char s_raw_buffer[];
    T* s_buffer = reinterpret_cast<T*>(s_raw_buffer);

    const size_t rope_offset = *d_rope_offset;
    const size_t head_idx = blockIdx.x;
    const size_t seq_idx = blockIdx.y;

    if (seq_idx >= seq_len) {
        return;
    }

    const size_t head_dim_half = head_dim / 2;
    const size_t group_idx = threadIdx.x;
    const size_t absolute_seq_pos = seq_idx + rope_offset;

    // Pointer to the current head in the Q input/output tensor
    T* q_head_ptr = q_inout + seq_idx * q_stride_s + head_idx * head_dim;

    const T* k_head_ptr = nullptr;
    T* k_cache_ptr = nullptr;
    const T* v_head_ptr = nullptr;
    T* v_cache_ptr = nullptr;

    // Setup pointers for K and V, and start async copy of V to shared memory
    if (head_idx < n_kv_heads) {
        k_head_ptr = k_input + seq_idx * k_stride_s + head_idx * head_dim;
        v_head_ptr = v_input + seq_idx * v_stride_s + head_idx * head_dim;

        const size_t cache_seq_pos = rope_offset + seq_idx;
        k_cache_ptr = k_cache_layer_ptr + cache_seq_pos * kv_cache_seq_stride + head_idx * head_dim;
        v_cache_ptr = v_cache_layer_ptr + cache_seq_pos * kv_cache_seq_stride + head_idx * head_dim;

        // Asynchronously copy V from global to shared memory for faster access
        unsigned int s_buffer_base_addr = __cvta_generic_to_shared(s_raw_buffer);
        constexpr int copy_bytes = 16;
        int copy_chunks = (head_dim * sizeof(T)) / copy_bytes;
        for (int i = threadIdx.x; i < copy_chunks; i += blockDim.x) {
            unsigned int dst_addr = s_buffer_base_addr + i * copy_bytes;
            CP_ASYNC_CA(dst_addr, v_head_ptr + (i * copy_bytes / sizeof(T)), copy_bytes);
        }
        CP_ASYNC_COMMIT_GROUP();
    }

    // Apply RoPE transformations to Q and K
    for (int i = 0; i < actual_pairs_per_thread; ++i) {
        size_t rot_dim = group_idx * actual_pairs_per_thread + i;
        if (rot_dim < head_dim_half) {
            size_t cache_idx = absolute_seq_pos * sin_cos_stride + rot_dim * 2;
            float2 sincos = *reinterpret_cast<const float2 *>(sin_cos_cache + cache_idx);

            // Apply to Q for all heads
            if (head_idx < n_heads) {
                float q0 = static_cast<float>(q_head_ptr[rot_dim]);
                float q1 = static_cast<float>(q_head_ptr[rot_dim + head_dim_half]);
                q_head_ptr[rot_dim] = static_cast<T>(q0 * sincos.y - q1 * sincos.x);
                q_head_ptr[rot_dim + head_dim_half] = static_cast<T>(q0 * sincos.x + q1 * sincos.y);
            }

            // Apply to K and write directly to cache (only for KV heads)
            if (k_cache_ptr) {
                float k0 = static_cast<float>(k_head_ptr[rot_dim]);
                float k1 = static_cast<float>(k_head_ptr[rot_dim + head_dim_half]);
                k_cache_ptr[rot_dim] = static_cast<T>(k0 * sincos.y - k1 * sincos.x);
                k_cache_ptr[rot_dim + head_dim_half] = static_cast<T>(k0 * sincos.x + k1 * sincos.y);
            }
        }
    }

    // Write V from shared memory to global cache
    if (v_cache_ptr) {
        CP_ASYNC_WAIT_ALL(); // Wait for the async copy to finish
        __syncthreads();

        // Use vectorized writes for performance
        constexpr int vec_size = sizeof(float4) / sizeof(T);
        if (head_dim % vec_size == 0) {
            const float4* s_v_vec_ptr = reinterpret_cast<const float4*>(s_buffer);
            float4* v_cache_vec_ptr = reinterpret_cast<float4*>(v_cache_ptr);
            const size_t head_dim_vec = head_dim / vec_size;
            for (size_t i = threadIdx.x; i < head_dim_vec; i += blockDim.x) {
                v_cache_vec_ptr[i] = s_v_vec_ptr[i];
            }
        } else { // Fallback for non-vectorized sizes
            for (size_t dim_idx = threadIdx.x; dim_idx < head_dim; dim_idx += blockDim.x) {
                v_cache_ptr[dim_idx] = s_buffer[dim_idx];
            }
        }
    }
}

// Wrapper function for the fused RoPE and KV cache kernel.
// This function simplifies the kernel launch by extracting parameters from Tensor objects.
template <typename T>
void rope_kv_fused(
    Tensor<T>& q_tensor,         // Input/Output Q tensor, shape [seq_len, n_heads, head_dim]
    const Tensor<T>& k_tensor,   // Input K tensor, shape [seq_len, n_kv_heads, head_dim]
    const Tensor<T>& v_tensor,   // Input V tensor, shape [seq_len, n_kv_heads, head_dim]
    Tensor<T>& k_cache_layer,    // K cache for a single layer, shape [max_len, n_kv_heads * head_dim]
    Tensor<T>& v_cache_layer,    // V cache for a single layer, shape [max_len, n_kv_heads * head_dim]
    const size_t* d_rope_offset,
    const Tensor<float>& sin_cos_cache,
    cudaStream_t stream
) {
    // --- Input Validation ---
    if (q_tensor.device() != Device::CUDA || k_tensor.device() != Device::CUDA || v_tensor.device() != Device::CUDA ||
        k_cache_layer.device() != Device::CUDA || v_cache_layer.device() != Device::CUDA || sin_cos_cache.device() != Device::CUDA) {
        throw std::runtime_error("rope_kv_fused: All tensors must be on CUDA device.");
    }
    
    const auto& q_sizes = q_tensor.sizes();
    const auto& k_sizes = k_tensor.sizes();
    const auto& v_sizes = v_tensor.sizes();

    if (q_sizes.size() != 3 || k_sizes.size() != 3 || v_sizes.size() != 3) {
        throw std::runtime_error("rope_kv_fused: Input tensors Q, K, V must be 3D.");
    }

    const size_t seq_len = q_sizes[0];
    const size_t n_heads = q_sizes[1];
    const size_t head_dim = q_sizes[2];
    const size_t n_kv_heads = k_sizes[1];

    if (seq_len == 0) return; // Nothing to do
    if (head_dim % 2 != 0) throw std::runtime_error("head_dim must be even for RoPE.");
    if (k_sizes[0] != seq_len || v_sizes[0] != seq_len || k_sizes[2] != head_dim || v_sizes[2] != head_dim) {
        throw std::runtime_error("rope_kv_fused: Q, K, V tensor dimensions are inconsistent.");
    }

    // --- Kernel Configuration ---
    constexpr int actual_pairs_per_thread = 2;
    size_t head_dim_half = head_dim / 2;
    int threads_per_block = (head_dim_half + actual_pairs_per_thread - 1) / actual_pairs_per_thread;
    threads_per_block = std::max(threads_per_block, (int)((head_dim + 3) / 4)); // Ensure enough threads for V copy
    threads_per_block = std::min(threads_per_block, 1024); // Adhere to CUDA limits

    dim3 grid_dim(n_heads, seq_len); // Grid covers all Q heads and sequence elements
    dim3 block_dim(threads_per_block);
    size_t smem_size = head_dim * sizeof(T); // Shared memory for one V head

    // --- Kernel Launch ---
    rope_kv_fused_kernel<T, actual_pairs_per_thread><<<grid_dim, block_dim, smem_size, stream>>>(
        q_tensor.data_ptr(),
        k_tensor.data_ptr(),
        v_tensor.data_ptr(),
        k_cache_layer.data_ptr(),
        v_cache_layer.data_ptr(),
        seq_len,
        n_heads,
        n_kv_heads,
        head_dim,
        d_rope_offset,
        sin_cos_cache.data_ptr(),
        sin_cos_cache.strides()[0],
        k_cache_layer.strides()[0],
        q_tensor.strides()[0],
        k_tensor.strides()[0],
        v_tensor.strides()[0]
    );
    
    // --- Error Checking ---
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("rope_kv_fused kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }
}

// --- Template Instantiations ---
template void rope_kv_fused<float>(
    Tensor<float>& q_tensor, const Tensor<float>& k_tensor, const Tensor<float>& v_tensor,
    Tensor<float>& k_cache_layer, Tensor<float>& v_cache_layer,
    const size_t* d_rope_offset, const Tensor<float>& sin_cos_cache, cudaStream_t stream);

template void rope_kv_fused<__nv_bfloat16>(
    Tensor<__nv_bfloat16>& q_tensor, const Tensor<__nv_bfloat16>& k_tensor, const Tensor<__nv_bfloat16>& v_tensor,
    Tensor<__nv_bfloat16>& k_cache_layer, Tensor<__nv_bfloat16>& v_cache_layer,
    const size_t* d_rope_offset, const Tensor<float>& sin_cos_cache, cudaStream_t stream);

}  // namespace cuda_OP
