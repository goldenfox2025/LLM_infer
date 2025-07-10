#include <cmath>
#include <iostream>
#include <stdexcept>

#include "cudaOP.cuh"

namespace cuda_OP {

// RoPE + KV Cache写入融合kernel
// 对K张量执行RoPE并直接写入KV cache，对V张量直接写入KV cache
template <typename T, int actual_pairs_per_thread = 2>
__global__ void rope_k_precompute_with_write_kv_kernel(
    const T *k_input,           // 输入的K张量 [seq_len, n_kv_heads, head_dim]
    const T *v_input,           // 输入的V张量 [seq_len, n_kv_heads, head_dim]
    T **kv_cache_k_ptrs,        // K cache指针数组 [seq_len]，每个指向对应位置的K cache
    T **kv_cache_v_ptrs,        // V cache指针数组 [seq_len]，每个指向对应位置的V cache
    size_t seq_len,
    size_t n_kv_heads,
    size_t head_dim,
    const size_t *d_offset,     // RoPE offset
    const float *sin_cos_cache, // 预计算的sin/cos缓存
    size_t cache_stride,        // sin_cos_cache的stride
    int k_input_stride,         // K输入张量的stride
    int v_input_stride          // V输入张量的stride
) {
    // 从设备内存读取offset
    size_t offset = *d_offset;
    
    // 线程组织：blockIdx.x = head_idx, blockIdx.y = seq_idx
    size_t head_idx = blockIdx.x;
    size_t seq_idx = blockIdx.y;
    
    // 边界检查
    if (head_idx >= n_kv_heads || seq_idx >= seq_len) {
        return;
    }
    
    // RoPE作用于向量的前一半和后一半
    size_t head_dim_half = head_dim / 2;
    
    // 当前线程的职责
    size_t group_idx = threadIdx.x;
    // Token在原始完整序列中的绝对位置
    size_t absolute_seq_pos = seq_idx + offset;
    
    // 输入张量的指针
    const T *k_head_ptr = k_input + seq_idx * k_input_stride + head_idx * head_dim;
    const T *v_head_ptr = v_input + seq_idx * v_input_stride + head_idx * head_dim;
    
    // 输出KV cache的指针
    T *k_cache_ptr = kv_cache_k_ptrs[seq_idx] + head_idx * head_dim;
    T *v_cache_ptr = kv_cache_v_ptrs[seq_idx] + head_idx * head_dim;
    
    // 执行K的RoPE变换并写入cache
    for (int i = 0; i < actual_pairs_per_thread; ++i) {
        // 当前线程要处理的旋转维度索引 (相对于 head_dim_half)
        size_t rot_dim = group_idx * actual_pairs_per_thread + i;
        
        // 边界检查：确保 rot_dim 没有超出当前头向量的一半长度
        if (rot_dim < head_dim_half) {
            // 从预计算缓存中读取sin/cos值
            size_t cache_idx = absolute_seq_pos * cache_stride + rot_dim * 2;
            float2 sincos = *reinterpret_cast<const float2 *>(sin_cos_cache + cache_idx);
            
            // 读取K张量的要旋转的一对数
            float x0 = static_cast<float>(k_head_ptr[rot_dim]);
            float x1 = static_cast<float>(k_head_ptr[rot_dim + head_dim_half]);
            
            // 执行RoPE变换并直接写入K cache
            k_cache_ptr[rot_dim] = static_cast<T>(x0 * sincos.y - x1 * sincos.x);
            k_cache_ptr[rot_dim + head_dim_half] = static_cast<T>(x0 * sincos.x + x1 * sincos.y);
        }
    }
    
    // V张量直接复制到cache（V不需要RoPE）
    for (size_t dim_idx = threadIdx.x; dim_idx < head_dim; dim_idx += blockDim.x) {
        v_cache_ptr[dim_idx] = v_head_ptr[dim_idx];
    }
}

// 简化版本的RoPE + KV Cache写入融合算子
template <typename T>
void rope_k_precompute_with_write_kv(
    const Tensor<T> &k_input,           // 输入的K张量 [seq_len, n_kv_heads, head_dim]
    const Tensor<T> &v_input,           // 输入的V张量 [seq_len, n_kv_heads, head_dim]
    const std::vector<Tensor<T>*> &k_cache_slices,  // K cache切片数组
    const std::vector<Tensor<T>*> &v_cache_slices,  // V cache切片数组
    const size_t *d_offset,             // RoPE offset
    const Tensor<float> *sin_cos_cache, // 预计算的sin/cos缓存
    cudaStream_t stream                 // CUDA stream
) {
    // 输入验证
    if (k_input.device() != Device::CUDA || v_input.device() != Device::CUDA) {
        throw std::runtime_error("rope_k_precompute_with_write_kv: Input tensors must be on CUDA device.");
    }
    if (d_offset == nullptr) {
        throw std::runtime_error("rope_k_precompute_with_write_kv: Device offset pointer cannot be null.");
    }
    if (sin_cos_cache == nullptr || sin_cos_cache->device() != Device::CUDA) {
        throw std::runtime_error("rope_k_precompute_with_write_kv: sin_cos_cache must be on CUDA device.");
    }
    
    // 解析张量维度
    const auto &k_sizes = k_input.sizes();
    const auto &v_sizes = v_input.sizes();
    
    if (k_sizes.size() != 3 || v_sizes.size() != 3) {
        throw std::runtime_error("rope_k_precompute_with_write_kv: Input tensors must be 3D (seq_len, n_kv_heads, head_dim).");
    }
    
    size_t seq_len = k_sizes[0];
    size_t n_kv_heads = k_sizes[1];
    size_t head_dim = k_sizes[2];
    
    // 验证K和V张量形状一致
    if (v_sizes[0] != seq_len || v_sizes[1] != n_kv_heads || v_sizes[2] != head_dim) {
        throw std::runtime_error("rope_k_precompute_with_write_kv: K and V tensors must have the same shape.");
    }
    
    // 验证cache切片数量
    if (k_cache_slices.size() != seq_len || v_cache_slices.size() != seq_len) {
        throw std::runtime_error("rope_k_precompute_with_write_kv: Cache slices count mismatch.");
    }
    
    // 验证缓存维度
    const auto &cache_sizes = sin_cos_cache->sizes();
    if (cache_sizes.size() != 2 || cache_sizes[1] != head_dim) {
        throw std::runtime_error("rope_k_precompute_with_write_kv: sin_cos_cache dimension mismatch.");
    }
    
    // 基本检查
    if (seq_len == 0 || n_kv_heads == 0 || head_dim == 0) {
        return;
    }
    if (head_dim % 2 != 0) {
        throw std::runtime_error("rope_k_precompute_with_write_kv: head_dim must be even.");
    }
    
    size_t head_dim_half = head_dim / 2;
    if (head_dim_half == 0) {
        return;
    }
    
    // 准备KV cache指针数组
    std::vector<T*> h_k_ptrs(seq_len);
    std::vector<T*> h_v_ptrs(seq_len);

    for (size_t j = 0; j < seq_len; j++) {
        h_k_ptrs[j] = k_cache_slices[j]->data_ptr();
        h_v_ptrs[j] = v_cache_slices[j]->data_ptr();
    }
    
    // 分配并复制指针数组到设备端
    T **d_k_ptrs = nullptr;
    T **d_v_ptrs = nullptr;
    
    cudaError_t err = cudaMalloc(&d_k_ptrs, seq_len * sizeof(T*));
    if (err != cudaSuccess) {
        throw std::runtime_error("rope_k_precompute_with_write_kv: Failed to allocate device memory for K pointers.");
    }
    
    err = cudaMalloc(&d_v_ptrs, seq_len * sizeof(T*));
    if (err != cudaSuccess) {
        cudaFree(d_k_ptrs);
        throw std::runtime_error("rope_k_precompute_with_write_kv: Failed to allocate device memory for V pointers.");
    }
    
    err = cudaMemcpyAsync(d_k_ptrs, h_k_ptrs.data(), seq_len * sizeof(T*), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        cudaFree(d_k_ptrs);
        cudaFree(d_v_ptrs);
        throw std::runtime_error("rope_k_precompute_with_write_kv: Failed to copy K pointers to device.");
    }
    
    err = cudaMemcpyAsync(d_v_ptrs, h_v_ptrs.data(), seq_len * sizeof(T*), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        cudaFree(d_k_ptrs);
        cudaFree(d_v_ptrs);
        throw std::runtime_error("rope_k_precompute_with_write_kv: Failed to copy V pointers to device.");
    }
    
    // 线程配置
    constexpr int actual_pairs_per_thread = 2;
    int threads_per_block_dim = (head_dim_half + actual_pairs_per_thread - 1) / actual_pairs_per_thread;
    
    // 确保线程数不超过1024，并且能够处理V的复制
    threads_per_block_dim = max(threads_per_block_dim, (int)((head_dim + 31) / 32));
    if (threads_per_block_dim > 1024) {
        threads_per_block_dim = 1024;
    }
    
    // Grid维度：每个(head, seq) 对应一个Block
    dim3 grid_dim(n_kv_heads, seq_len);
    // Block维度：用于处理RoPE和V复制的线程数
    dim3 block_dim(threads_per_block_dim);
    
    // 启动融合kernel
    int k_input_stride = k_input.strides()[0];
    int v_input_stride = v_input.strides()[0];
    size_t cache_stride = sin_cos_cache->sizes()[1];
    
    rope_k_precompute_with_write_kv_kernel<T, actual_pairs_per_thread><<<grid_dim, block_dim, 0, stream>>>(
        k_input.data_ptr(),
        v_input.data_ptr(),
        d_k_ptrs,
        d_v_ptrs,
        seq_len,
        n_kv_heads,
        head_dim,
        d_offset,
        sin_cos_cache->data_ptr(),
        cache_stride,
        k_input_stride,
        v_input_stride
    );
    
    // 清理设备端指针数组（异步清理）
    cudaStreamAddCallback(stream, [](cudaStream_t stream, cudaError_t status, void *userData) {
        T **ptrs = static_cast<T**>(userData);
        cudaFree(ptrs);
    }, d_k_ptrs, 0);
    
    cudaStreamAddCallback(stream, [](cudaStream_t stream, cudaError_t status, void *userData) {
        T **ptrs = static_cast<T**>(userData);
        cudaFree(ptrs);
    }, d_v_ptrs, 0);
    
    // 错误检查
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error after rope_k_precompute_with_write_kv kernel launch: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("rope_k_precompute_with_write_kv CUDA kernel launch failed");
    }
}

// 模板实例化
template void rope_k_precompute_with_write_kv<float>(
    const Tensor<float> &k_input,
    const Tensor<float> &v_input,
    const std::vector<Tensor<float>*> &k_cache_slices,
    const std::vector<Tensor<float>*> &v_cache_slices,
    const size_t *d_offset,
    const Tensor<float> *sin_cos_cache,
    cudaStream_t stream
);

template void rope_k_precompute_with_write_kv<__nv_bfloat16>(
    const Tensor<__nv_bfloat16> &k_input,
    const Tensor<__nv_bfloat16> &v_input,
    const std::vector<Tensor<__nv_bfloat16>*> &k_cache_slices,
    const std::vector<Tensor<__nv_bfloat16>*> &v_cache_slices,
    const size_t *d_offset,
    const Tensor<float> *sin_cos_cache,
    cudaStream_t stream
);

}  // namespace cuda_OP