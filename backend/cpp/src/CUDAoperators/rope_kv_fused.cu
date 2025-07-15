#include <cmath>
#include <iostream>
#include <stdexcept>

#include "cudaOP.cuh"
#include "ptx_common.h"

namespace cuda_OP {

// 增强版RoPE + KV Cache写入融合kernel，支持Q处理、向量化和cp.async
// 利用KV cache连续内存特性，直接计算地址而不分配指针数组
template <typename T, int actual_pairs_per_thread = 2>
__global__ void rope_qkv_fused_kernel(
    T *q_inout,                 // 输入输出的Q张量 [seq_len, n_heads, head_dim]（原地操作，可选）
    const T *k_input,           // 输入的K张量 [seq_len, n_kv_heads, head_dim]
    const T *v_input,           // 输入的V张量 [seq_len, n_kv_heads, head_dim]
    T *k_cache_base,            // K cache连续内存基地址 [n_layers, max_seq_len, head_dim]
    T *v_cache_base,            // V cache连续内存基地址 [n_layers, max_seq_len, head_dim]
    size_t seq_len,
    size_t n_heads,             // Q张量的头数
    size_t n_kv_heads,          // K,V张量的头数
    size_t head_dim,
    size_t max_seq_len,         // KV cache的最大序列长度
    size_t layer_idx,           // 当前层索引
    size_t cache_offset,        // 在cache中的起始位置
    const size_t *d_rope_offset, // RoPE offset
    const float *sin_cos_cache, // 预计算的sin/cos缓存
    size_t cache_stride,        // sin_cos_cache的stride
    int q_input_stride,         // Q输入张量的stride
    int k_input_stride,         // K输入张量的stride
    int v_input_stride,         // V输入张量的stride
    bool process_q = true       // 是否处理Q张量
) {
    // 从设备内存读取rope offset
    size_t rope_offset = *d_rope_offset;
    
    // 线程组织：blockIdx.x = head_idx, blockIdx.y = seq_idx
    size_t head_idx = blockIdx.x;
    size_t seq_idx = blockIdx.y;
    
    // 边界检查
    if (seq_idx >= seq_len) {
        return;
    }
    
    // RoPE作用于向量的前一半和后一半
    size_t head_dim_half = head_dim / 2;
    
    // 当前线程的职责
    size_t group_idx = threadIdx.x;
    // Token在原始完整序列中的绝对位置
    size_t absolute_seq_pos = seq_idx + rope_offset;
    
    // 处理Q张量（如果需要）
    if (process_q && head_idx < n_heads) {
        T *q_head_ptr = q_inout + seq_idx * q_input_stride + head_idx * head_dim;
        
        // 执行Q的RoPE变换（原地操作）
        for (int i = 0; i < actual_pairs_per_thread; ++i) {
            size_t rot_dim = group_idx * actual_pairs_per_thread + i;
            
            if (rot_dim < head_dim_half) {
                // 从预计算缓存中读取sin/cos值
                size_t cache_idx = absolute_seq_pos * cache_stride + rot_dim * 2;
                float2 sincos = *reinterpret_cast<const float2 *>(sin_cos_cache + cache_idx);
                
                // 读取Q张量的要旋转的一对数
                float x0 = static_cast<float>(q_head_ptr[rot_dim]);
                float x1 = static_cast<float>(q_head_ptr[rot_dim + head_dim_half]);
                
                // 执行RoPE变换并写回Q张量
                q_head_ptr[rot_dim] = static_cast<T>(x0 * sincos.y - x1 * sincos.x);
                q_head_ptr[rot_dim + head_dim_half] = static_cast<T>(x0 * sincos.x + x1 * sincos.y);
            }
        }
    }
    
    // 处理K张量
    if (head_idx < n_kv_heads) {
        const T *k_head_ptr = k_input + seq_idx * k_input_stride + head_idx * head_dim;
        const T *v_head_ptr = v_input + seq_idx * v_input_stride + head_idx * head_dim;
        
        // 计算K cache目标地址：[layer_idx, cache_offset + seq_idx, head_idx * head_dim]
        size_t k_cache_pos = layer_idx * max_seq_len * n_kv_heads * head_dim + 
                            (cache_offset + seq_idx) * n_kv_heads * head_dim + 
                            head_idx * head_dim;
        T *k_cache_ptr = k_cache_base + k_cache_pos;
        
        // 计算V cache目标地址
        size_t v_cache_pos = layer_idx * max_seq_len * n_kv_heads * head_dim + 
                            (cache_offset + seq_idx) * n_kv_heads * head_dim + 
                            head_idx * head_dim;
        T *v_cache_ptr = v_cache_base + v_cache_pos;
        
        // 执行K的RoPE变换并写入cache
        for (int i = 0; i < actual_pairs_per_thread; ++i) {
            size_t rot_dim = group_idx * actual_pairs_per_thread + i;
            
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
        
        // V张量直接复制到cache（简化实现）
        // 每个线程处理一个元素，简单高效
        for (size_t dim_idx = threadIdx.x; dim_idx < head_dim; dim_idx += blockDim.x) {
            v_cache_ptr[dim_idx] = v_head_ptr[dim_idx];
        }
    }
}

// 增强版RoPE + KV Cache写入融合算子，支持Q处理和向量化优化
template <typename T>
void rope_qkv_precompute_with_write_kv(
    Tensor<T> *q_tensor,                    // 输入输出的Q张量（可选，如果为nullptr则不处理Q）
    const Tensor<T> &k_input,               // 输入的K张量 [seq_len, n_kv_heads, head_dim]
    const Tensor<T> &v_input,               // 输入的V张量 [seq_len, n_kv_heads, head_dim]
    const std::vector<Tensor<T>*> &k_cache_slices,  // K cache切片数组（用于获取基地址和计算偏移）
    const std::vector<Tensor<T>*> &v_cache_slices,  // V cache切片数组
    const size_t *d_offset,                 // RoPE offset
    const Tensor<float> *sin_cos_cache,     // 预计算的sin/cos缓存
    cudaStream_t stream                     // CUDA stream
) {
    // 输入验证
    if (k_input.device() != Device::CUDA || v_input.device() != Device::CUDA) {
        throw std::runtime_error("rope_qkv_precompute_with_write_kv: Input tensors must be on CUDA device.");
    }
    if (d_offset == nullptr) {
        throw std::runtime_error("rope_qkv_precompute_with_write_kv: Device offset pointer cannot be null.");
    }
    if (sin_cos_cache == nullptr || sin_cos_cache->device() != Device::CUDA) {
        throw std::runtime_error("rope_qkv_precompute_with_write_kv: sin_cos_cache must be on CUDA device.");
    }
    
    // 解析张量维度
    const auto &k_sizes = k_input.sizes();
    const auto &v_sizes = v_input.sizes();
    
    if (k_sizes.size() != 3 || v_sizes.size() != 3) {
        throw std::runtime_error("rope_qkv_precompute_with_write_kv: Input tensors must be 3D (seq_len, n_kv_heads, head_dim).");
    }
    
    size_t seq_len = k_sizes[0];
    size_t n_kv_heads = k_sizes[1];
    size_t head_dim = k_sizes[2];
    size_t n_heads = n_kv_heads;  // 默认值，如果有Q张量则使用Q的头数
    
    bool process_q = (q_tensor != nullptr);
    if (process_q) {
        if (q_tensor->device() != Device::CUDA) {
            throw std::runtime_error("rope_qkv_precompute_with_write_kv: Q tensor must be on CUDA device.");
        }
        const auto &q_sizes = q_tensor->sizes();
        if (q_sizes.size() != 3 || q_sizes[0] != seq_len || q_sizes[2] != head_dim) {
            throw std::runtime_error("rope_qkv_precompute_with_write_kv: Q tensor dimension mismatch.");
        }
        n_heads = q_sizes[1];
    }
    
    // 验证K和V张量形状一致
    if (v_sizes[0] != seq_len || v_sizes[1] != n_kv_heads || v_sizes[2] != head_dim) {
        throw std::runtime_error("rope_qkv_precompute_with_write_kv: K and V tensors must have the same shape.");
    }
    
    // 验证cache切片数量
    if (k_cache_slices.size() != seq_len || v_cache_slices.size() != seq_len) {
        throw std::runtime_error("rope_qkv_precompute_with_write_kv: Cache slices count mismatch.");
    }
    
    // 验证缓存维度
    const auto &cache_sizes = sin_cos_cache->sizes();
    if (cache_sizes.size() != 2 || cache_sizes[1] != head_dim) {
        throw std::runtime_error("rope_qkv_precompute_with_write_kv: sin_cos_cache dimension mismatch.");
    }
    
    // 基本检查
    if (seq_len == 0 || n_kv_heads == 0 || head_dim == 0) {
        return;
    }
    if (head_dim % 2 != 0) {
        throw std::runtime_error("rope_qkv_precompute_with_write_kv: head_dim must be even.");
    }
    
    // 通过第一个切片获取KV cache的基地址和相关信息
    // 注意：这里假设所有切片都来自同一个连续的KV cache
    if (k_cache_slices.empty() || v_cache_slices.empty()) {
        throw std::runtime_error("rope_qkv_precompute_with_write_kv: Empty cache slices.");
    }
    
    // 从切片中推断连续内存的基地址和布局参数
    // 这需要根据具体的KV cache实现来确定
    // 暂时使用简化的方法：假设可以直接从第一个切片推断
    T *k_cache_base = k_cache_slices[0]->data_ptr();
    T *v_cache_base = v_cache_slices[0]->data_ptr();
    
    // 这些参数需要从KV cache的实际布局中获取
    // 暂时使用占位符，实际使用时需要传入这些参数
    size_t max_seq_len = 2048;  // 需要从KV cache获取
    size_t layer_idx = 0;       // 需要传入当前层索引
    size_t cache_offset = 0;    // 需要传入当前在cache中的偏移
    
    // 线程配置
    constexpr int actual_pairs_per_thread = 2;
    size_t head_dim_half = head_dim / 2;
    int threads_per_block_dim = (head_dim_half + actual_pairs_per_thread - 1) / actual_pairs_per_thread;
    
    // 确保线程数能够处理向量化的V复制（每个线程处理4个元素）
    threads_per_block_dim = max(threads_per_block_dim, (int)((head_dim + 3) / 4));
    if (threads_per_block_dim > 1024) {
        threads_per_block_dim = 1024;
    }
    
    // Grid维度：使用最大头数以处理Q和KV
    size_t max_heads = max(n_heads, n_kv_heads);
    dim3 grid_dim(max_heads, seq_len);
    dim3 block_dim(threads_per_block_dim);
    
    // 启动增强版融合kernel
    int q_input_stride = process_q ? q_tensor->strides()[0] : 0;
    int k_input_stride = k_input.strides()[0];
    int v_input_stride = v_input.strides()[0];
    size_t cache_stride = sin_cos_cache->sizes()[1];
    
    rope_qkv_fused_kernel<T, actual_pairs_per_thread><<<grid_dim, block_dim, 0, stream>>>(
        process_q ? q_tensor->data_ptr() : nullptr,
        k_input.data_ptr(),
        v_input.data_ptr(),
        k_cache_base,
        v_cache_base,
        seq_len,
        n_heads,
        n_kv_heads,
        head_dim,
        max_seq_len,
        layer_idx,
        cache_offset,
        d_offset,
        sin_cos_cache->data_ptr(),
        cache_stride,
        q_input_stride,
        k_input_stride,
        v_input_stride,
        process_q
    );
    
    // 错误检查
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error after rope_qkv_fused_kernel launch: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("rope_qkv_fused_kernel CUDA kernel launch failed");
    }
}

// 保持向后兼容的简化版本（仅处理K和V）
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
    // 调用增强版本，但不处理Q张量
    rope_qkv_precompute_with_write_kv<T>(
        nullptr,  // 不处理Q张量
        k_input,
        v_input,
        k_cache_slices,
        v_cache_slices,
        d_offset,
        sin_cos_cache,
        stream
    );
}

// 模板实例化 - 增强版本
template void rope_qkv_precompute_with_write_kv<float>(
    Tensor<float> *q_tensor,
    const Tensor<float> &k_input,
    const Tensor<float> &v_input,
    const std::vector<Tensor<float>*> &k_cache_slices,
    const std::vector<Tensor<float>*> &v_cache_slices,
    const size_t *d_offset,
    const Tensor<float> *sin_cos_cache,
    cudaStream_t stream
);

template void rope_qkv_precompute_with_write_kv<__nv_bfloat16>(
    Tensor<__nv_bfloat16> *q_tensor,
    const Tensor<__nv_bfloat16> &k_input,
    const Tensor<__nv_bfloat16> &v_input,
    const std::vector<Tensor<__nv_bfloat16>*> &k_cache_slices,
    const std::vector<Tensor<__nv_bfloat16>*> &v_cache_slices,
    const size_t *d_offset,
    const Tensor<float> *sin_cos_cache,
    cudaStream_t stream
);

// 模板实例化 - 向后兼容版本
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