#include "cudaOP.cuh"

namespace cuda_OP {

template <typename T, int WARP_SIZE = 32>
__global__ void gemv_qkv_kernel(const T *A, const T *B, T *q, T *k, T *v, const T *bias, 
                                int *offset_array, int layer_index, int N, int K, 
                                int Q_len, int K_len, int V_len, int n_layers,
                                int *pingpong_index) {
    
    // 从offset数组中获取当前层的偏移值
    int out_off = 0;
    if (offset_array != nullptr) {
        out_off = offset_array[layer_index + n_layers * (*pingpong_index)];
    }

    const int tid_x = threadIdx.x;
    const int tid_y = threadIdx.y;
    const int global_n = blockIdx.x * blockDim.y + tid_y;  // 这是输出向量的索引 (0 to N-1)

    if (global_n >= N) {
        return;
    }

    constexpr int VEC_UNIT = sizeof(float4) / sizeof(T);
    float acc = 0;

    // 每个warp负责计算B的一行与A的点积
    const T *b_row_ptr = B + global_n * K;

    for (int k_idx = tid_x * VEC_UNIT; k_idx < K; k_idx += WARP_SIZE * VEC_UNIT) {
        // 边界检查
        if (k_idx + VEC_UNIT <= K) {
            Vec<T, VEC_UNIT> va, vb;
            // A是共享的输入向量，所有warp都从中读取相同的数据
            va.f4 = *reinterpret_cast<const float4 *>(A + k_idx);
            vb.f4 = *reinterpret_cast<const float4 *>(b_row_ptr + k_idx);

// 点积计算
#pragma unroll
            for (int j = 0; j < VEC_UNIT; ++j) {
                acc += static_cast<float>(va.t[j]) * static_cast<float>(vb.t[j]);
            }
        } else {
            for (int j = k_idx; j < K; ++j) {
                acc += static_cast<float>(A[j]) * static_cast<float>(b_row_ptr[j]);
            }
        }
    }

    // Warp-level reduction
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xffffffff, acc, offset);
    }

    // 只有每个warp的第一个线程 (tid_x == 0) 写入结果
    if (tid_x == 0) {
        if (bias != nullptr) {
            acc += static_cast<float>(bias[global_n]);
        }

        if (global_n < Q_len) {
            T *out_ptr = q + global_n;
            *out_ptr = static_cast<T>(acc);
        } else if (global_n < Q_len + K_len) {
            T *out_ptr = k + out_off + (global_n - Q_len);
            *out_ptr = static_cast<T>(acc);
        } else if (global_n < Q_len + K_len + V_len) {
            T *out_ptr = v + out_off + (global_n - Q_len - K_len);
            *out_ptr = static_cast<T>(acc);
        }
    }
}

// 专门的RoPE kernel，使用与原版相同的线程组织方式
template <typename T, int actual_pairs_per_thread = 2>
__global__ void rope_kernel_fused(T *q_tensor, T *k_tensor, size_t seq_len, size_t n_heads, size_t n_kv_heads,
                                  size_t head_dim, const size_t *d_offset, const float *sin_cos_cache,
                                  size_t cache_stride, int q_stride, int k_stride, int *offset_array, 
                                  int layer_index, int n_layers, int *pingpong_index) {
    size_t offset;
    
    // 从设备内存读取offset
    if (pingpong_index != nullptr) {
        offset = d_offset[*pingpong_index];
    } else {
        offset = d_offset[0];
    }

    int t_off = 0;
    if (offset_array != nullptr) {
        t_off = offset_array[layer_index + n_layers * (*pingpong_index)];
    }

    // 线程组织：blockIdx.x = tensor_type (0=Q, 1=K), blockIdx.y = head_idx, blockIdx.z = seq_idx
    size_t tensor_type = blockIdx.x;  // 0 for Q, 1 for K
    size_t head_idx = blockIdx.y;
    size_t seq_idx_in_batch = blockIdx.z;

    // RoPE作用于向量的前一半和后一半
    size_t head_dim_half = head_dim / 2;

    // 当前线程的职责
    size_t group_idx = threadIdx.x;
    // Token在原始完整序列中的绝对位置
    size_t absolute_seq_pos = seq_idx_in_batch + offset;

    T *current_head_ptr;
    
    if (tensor_type == 0) {
        // Q tensor
        if (head_idx >= n_heads) return;
        current_head_ptr = q_tensor + seq_idx_in_batch * q_stride + head_idx * head_dim;
    } else {
        // K tensor  
        if (head_idx >= n_kv_heads) return;
        current_head_ptr = k_tensor + t_off + seq_idx_in_batch * k_stride + head_idx * head_dim;
    }

    // 执行旋转 - 与原版完全相同的逻辑
    for (int i = 0; i < actual_pairs_per_thread; ++i) {
        // 当前线程要处理的旋转维度索引 (相对于 head_dim_half)
        size_t rot_dim = group_idx * actual_pairs_per_thread + i;

        // 边界检查：确保 rot_dim 没有超出当前头向量的一半长度
        if (rot_dim < head_dim_half) {
            size_t cache_idx = absolute_seq_pos * cache_stride + rot_dim * 2;
            float2 sincos = *reinterpret_cast<const float2 *>(sin_cos_cache + cache_idx);

            float x0 = static_cast<float>(current_head_ptr[rot_dim]);
            float x1 = static_cast<float>(current_head_ptr[rot_dim + head_dim_half]);
            current_head_ptr[rot_dim] = static_cast<T>(x0 * sincos.y - x1 * sincos.x);
            current_head_ptr[rot_dim + head_dim_half] = static_cast<T>(x0 * sincos.x + x1 * sincos.y);
        }
    }
}

template <typename T>
void gemv_qkv_rope(const Tensor<T> *A, const Tensor<T> *B, Tensor<T> *q, Tensor<T> *k, Tensor<T> *v, 
                   const Tensor<T> *bias, const size_t *d_offset, const Tensor<float> *sin_cos_cache,
                   int *offset_array, int layer_index, size_t Q_len, size_t K_len, size_t V_len,
                   size_t n_heads, size_t n_kv_heads, size_t head_dim,
                   cudaStream_t stream, int n_layers, int *pingpong_index) {
    
    // Input validation
    if (!A || !B || !q || !k || !v || !d_offset || !sin_cos_cache) {
        throw std::runtime_error("gemv_qkv_rope: null pointer arguments");
    }
    
    if (head_dim % 2 != 0) {
        throw std::runtime_error("gemv_qkv_rope: head_dim must be even for RoPE");
    }

    const int N = B->sizes()[1];  // Total output dimension (Q_len + K_len + V_len)
    const int K = B->sizes()[0];  // Input dimension

    const T *d_A = A->data_ptr();
    const T *d_B = B->data_ptr();
    T *d_q = q->data_ptr();
    T *d_k = k->data_ptr();
    T *d_v = v->data_ptr();
    const T *d_bias = bias ? bias->data_ptr() : nullptr;

    // Step 1: GEMV QKV computation
    constexpr int ROWS_PER_BLOCK = 4;
    dim3 blockDim(32, ROWS_PER_BLOCK);
    dim3 gridDim((N + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, 1);

    gemv_qkv_kernel<T><<<gridDim, blockDim, 0, stream>>>(
        d_A, d_B, d_q, d_k, d_v, d_bias, offset_array, layer_index, N, K, 
        Q_len, K_len, V_len, n_layers, pingpong_index);

    // Step 2: Apply RoPE to Q and K tensors using separate kernel with original logic
    const auto &cache_sizes = sin_cos_cache->sizes();
    size_t cache_stride = cache_sizes[1];  // head_dim
    
    int q_stride = q->strides()[0];
    int k_stride = k->strides()[0];
    
    size_t seq_len = 1;  // For graph mode, always 1
    size_t head_dim_half = head_dim / 2;
    
    // 使用与原版RoPE相同的线程配置
    constexpr int actual_pairs_per_thread = 2;
    int threads_per_block_dim = (head_dim_half + actual_pairs_per_thread - 1) / actual_pairs_per_thread;
    
    if (threads_per_block_dim > 1024) {
        throw std::runtime_error("gemv_qkv_rope: head_dim too large for RoPE kernel");
    }

    // Grid: (2, max_heads, seq_len) - 2 for Q and K tensors
    size_t max_heads = max(n_heads, n_kv_heads);
    dim3 rope_grid_dim(2, max_heads, seq_len);
    dim3 rope_block_dim(threads_per_block_dim);

    rope_kernel_fused<T><<<rope_grid_dim, rope_block_dim, 0, stream>>>(
        d_q, d_k, seq_len, n_heads, n_kv_heads, head_dim, d_offset, sin_cos_cache->data_ptr(),
        cache_stride, q_stride, k_stride, offset_array, layer_index, n_layers, pingpong_index);

    // Error checking
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("gemv_qkv_rope kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }
}

// Explicit template instantiations
template void gemv_qkv_rope<nv_bfloat16>(const Tensor<nv_bfloat16> *A, const Tensor<nv_bfloat16> *B, 
                                         Tensor<nv_bfloat16> *q, Tensor<nv_bfloat16> *k, Tensor<nv_bfloat16> *v,
                                         const Tensor<nv_bfloat16> *bias, const size_t *d_offset, const Tensor<float> *sin_cos_cache,
                                         int *offset_array, int layer_index, size_t Q_len, size_t K_len, size_t V_len,
                                         size_t n_heads, size_t n_kv_heads, size_t head_dim,
                                         cudaStream_t stream, int n_layers, int *pingpong_index);

template void gemv_qkv_rope<float>(const Tensor<float> *A, const Tensor<float> *B, 
                                   Tensor<float> *q, Tensor<float> *k, Tensor<float> *v,
                                   const Tensor<float> *bias, const size_t *d_offset, const Tensor<float> *sin_cos_cache,
                                   int *offset_array, int layer_index, size_t Q_len, size_t K_len, size_t V_len,
                                   size_t n_heads, size_t n_kv_heads, size_t head_dim,
                                   cudaStream_t stream, int n_layers, int *pingpong_index);

}  // namespace cuda_OP