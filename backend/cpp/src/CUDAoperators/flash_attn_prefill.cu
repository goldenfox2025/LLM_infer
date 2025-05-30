#include <float.h>  // 用于 FLT_MAX

#include "cudaOP.cuh"

__device__ __forceinline__ float warpReduceSum(float val) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
  }
  return __shfl_sync(0xFFFFFFFF, val, 0);  // 将lane 0的结果广播到所有lane
}

__device__ __forceinline__ float warpReduceMax(float val) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    val = max(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
  }
  return __shfl_sync(0xFFFFFFFF, val, 0);  // 将lane 0的结果广播到所有lane
}

namespace cuda_OP {

// 模板参数含义:
// T: 数据类型 (例如 float, __half, __nv_bfloat16)
// B_c: 处理K/V时，每个K_j/V_j块包含的token数量 (列块大小)
// B_r: 处理Q时，每个Q_i块包含的token数量 (行块大小)
// T_r: 当前CUDA块负责处理的Q段（由divide_len定义）中的Q
// token总数。外层Q循环以此为上限。 T_c: K/V
// Cache的总token数上限（通常是K/V缓冲区的最大长度）。内层K/V循环以此为上限。
// divide_len: 定义了 blockIdx.x
// 对应的Q段的长度。q_segment_global_start_token_idx = blockIdx.x * divide_len。
// NUM_ELEM_PER_THREAD:
// 用户原始模板参数，此版S_ij实现中未直接使用其原始复杂逻辑，但保留参数以匹配风格。
// WARP_NUM: CUDA块中Warp的数量。
// DQKV: 注意力头的维度。
template <typename T, int B_c, int B_r, int T_r, int T_c, int divide_len,
          int NUM_ELEM_PER_THREAD, int WARP_NUM = 4,
          int DQKV = 128>
__global__ void flash_attn_prefill_kernel(
    const T* __restrict__ q_global,  // 全局Q张量指针
    const T* __restrict__ k_global,  // 全局K张量指针
    const T* __restrict__ v_global,  // 全局V张量指针
    T* __restrict__ out_global,      // 全局输出O张量指针
    int num_q_heads_total,           // Q头的总数 (n_q_h)
    int num_kv_heads_total,          // KV头的总数 (n_kv_h)
    int GQA_n_group,                 // GQA分组数 (n_group)
    int current_prefill_q_length,    // 当前prefill操作中，Q序列的实际总长度
    int current_kv_cache_total_len,  // K/V Cache当前的有效总长度
    int q_offset_in_kv_timeline      // Q在整个KV时间轴上的起始偏移量
) {
  // Q分段
  int q_segment_id_from_blockidx = blockIdx.x;
  int q_segment_global_start_token_idx =
      q_segment_id_from_blockidx * divide_len;

  // Qhead分块
  int q_head_idx_global = blockIdx.y;
  int kv_head_idx_global = q_head_idx_global / GQA_n_group;

  int tid_in_block = threadIdx.x;
  int warp_idx_in_block = tid_in_block / warpSize;
  int lane_idx_in_warp = tid_in_block % warpSize;

  // 共享内存
  extern __shared__ char shared_mem_char[];
  T* q_smem = reinterpret_cast<T*>(shared_mem_char);
  T* k_smem = reinterpret_cast<T*>((char*)q_smem + B_r * DQKV * sizeof(T));
  T* v_smem = reinterpret_cast<T*>((char*)k_smem + B_c * DQKV * sizeof(T));
  float* s_tmp =
      reinterpret_cast<float*>((char*)v_smem + B_c * DQKV * sizeof(T));
  T* o_smem = reinterpret_cast<T*>((char*)s_tmp + B_r * B_c * sizeof(float));
  float* m_stats =
      reinterpret_cast<float*>((char*)o_smem + B_r * DQKV * sizeof(T));
  float* l_stats =
      reinterpret_cast<float*>((char*)m_stats + B_r * sizeof(float));

  // 接下来开始计算
  // 遍历当前CUDA块负责的Q段（长度T_r）内的各个 Q_i 块（大小B_r）
  for (int i_q_chunk_offset_in_segment = 0; i_q_chunk_offset_in_segment < T_r;
       i_q_chunk_offset_in_segment += B_r) {
    for (int r_init_idx = tid_in_block; r_init_idx < B_r;
         r_init_idx += blockDim.x) {
      if (r_init_idx < B_r) {
        m_stats[r_init_idx] = -FLT_MAX;
        l_stats[r_init_idx] = 0.0f;
        for (int d_init_idx = 0; d_init_idx < DQKV; ++d_init_idx) {
          o_smem[r_init_idx * DQKV + d_init_idx] = static_cast<T>(0.0f);
        }
      }
    }
    __syncthreads();  // 确保初始化完成

    // 加载 Q_i 块 (B_r 行) 到共享内存 q_smem
    for (int r_local_q_idx = 0; r_local_q_idx < B_r; ++r_local_q_idx) {
      int current_q_token_relative_idx_in_prefill =
          q_segment_global_start_token_idx + i_q_chunk_offset_in_segment +
          r_local_q_idx;
      bool q_token_is_valid =
          (current_q_token_relative_idx_in_prefill < current_prefill_q_length);
      const T* q_global_row_ptr =
          q_global +
          (current_q_token_relative_idx_in_prefill * num_q_heads_total +
           q_head_idx_global) *
              DQKV;

      for (int d_load_idx = tid_in_block; d_load_idx < DQKV;
           d_load_idx += blockDim.x) {
        if (q_token_is_valid) {
          q_smem[r_local_q_idx * DQKV + d_load_idx] =
              q_global_row_ptr[d_load_idx];
        } else {
          q_smem[r_local_q_idx * DQKV + d_load_idx] = static_cast<T>(0.0f);
        }
      }
    }
    __syncthreads();  // 确保q_smem加载完毕

    // T_c是最大长度，实际迭代到 current_kv_cache_total_len
    // 即可，通过mask控制有效性
    for (int j_kv_chunk_start_token_idx = 0;
         j_kv_chunk_start_token_idx < current_kv_cache_total_len;
         j_kv_chunk_start_token_idx += B_c) {
      // --- 加载 K_j 块 (B_c 行) 到 k_smem ---
      for (int c_local_k_idx = 0; c_local_k_idx < B_c; ++c_local_k_idx) {
        int current_k_global_token_idx =
            j_kv_chunk_start_token_idx + c_local_k_idx;
        bool k_token_is_valid =
            (current_k_global_token_idx < current_kv_cache_total_len);

        const T* k_global_row_ptr =
            k_global + (current_k_global_token_idx * num_kv_heads_total +
                        kv_head_idx_global) *
                           DQKV;

        for (int d_load_idx = tid_in_block; d_load_idx < DQKV;
             d_load_idx += blockDim.x) {
          if (k_token_is_valid) {
            k_smem[c_local_k_idx * DQKV + d_load_idx] =
                k_global_row_ptr[d_load_idx];
          } else {
            k_smem[c_local_k_idx * DQKV + d_load_idx] = static_cast<T>(0.0f);
          }
        }
      }

      // --- 加载 V_j 块 (B_c 行) 到 v_smem ---
      for (int c_local_v_idx = 0; c_local_v_idx < B_c; ++c_local_v_idx) {
        int current_v_global_token_idx =
            j_kv_chunk_start_token_idx + c_local_v_idx;
        bool v_token_is_valid =
            (current_v_global_token_idx < current_kv_cache_total_len);

        const T* v_global_row_ptr =
            v_global + (current_v_global_token_idx * num_kv_heads_total +
                        kv_head_idx_global) *
                           DQKV;

        for (int d_load_idx = tid_in_block; d_load_idx < DQKV;
             d_load_idx += blockDim.x) {
          if (v_token_is_valid) {
            v_smem[c_local_v_idx * DQKV + d_load_idx] =
                v_global_row_ptr[d_load_idx];
          } else {
            v_smem[c_local_v_idx * DQKV + d_load_idx] = static_cast<T>(0.0f);
          }
        }
      }
      __syncthreads();  // 确保k_smem和v_smem加载完毕

      // --- 计算 S_ij = Q_i * K_j^T ---
      // NUM_Q_ROWS_PER_WARP: 每个Warp在此阶段负责处理的Q_i块中的行数
      // B_r (Q块的行数) WARP_NUM (块中的Warp数)
      // 例如 B_r=8, WARP_NUM=2 -> NUM_Q_ROWS_PER_WARP = 4
      // 这意味着块中的每个Warp，会负责计算Q_i块中B_r/WARP_NUM行的S_ij元素
      constexpr int NUM_Q_ROWS_PER_WARP = (B_r + WARP_NUM - 1) / WARP_NUM;

      for (int q_row_offset_in_warp = 0;
           q_row_offset_in_warp < NUM_Q_ROWS_PER_WARP; ++q_row_offset_in_warp) {
        int r_smem_q_idx =
            warp_idx_in_block * NUM_Q_ROWS_PER_WARP + q_row_offset_in_warp;

        if (r_smem_q_idx < B_r) {  // 确保实际处理的Q行在B_r范围内
          int q_token_relative_idx_in_prefill =
              q_segment_global_start_token_idx + i_q_chunk_offset_in_segment +
              r_smem_q_idx;
          int q_global_pos_for_causal =
              q_offset_in_kv_timeline + q_token_relative_idx_in_prefill;

          for (int c_smem_k_idx = 0; c_smem_k_idx < B_c; ++c_smem_k_idx) {
            float psum_for_s_cell = 0.0f;
            for (int d_dot = lane_idx_in_warp; d_dot < DQKV;
                 d_dot += warpSize) {
              psum_for_s_cell +=
                  static_cast<float>(q_smem[r_smem_q_idx * DQKV + d_dot]) *
                  static_cast<float>(k_smem[c_smem_k_idx * DQKV + d_dot]);
            }

            float final_s_cell_val =
                warpReduceSum(psum_for_s_cell);  // 结果已广播

            const float scale_factor_d =
                1.0f / sqrtf(static_cast<float>(DQKV));  // 计算一次缩放因子
            final_s_cell_val *= scale_factor_d;          // 应用缩放

            if (lane_idx_in_warp == 0) {  // Warp中一个线程写入结果并应用掩码
              int current_k_global_token_idx_for_s =
                  j_kv_chunk_start_token_idx + c_smem_k_idx;

              bool q_is_padding_for_s =
                  (q_token_relative_idx_in_prefill >= current_prefill_q_length);
              bool k_is_padding_for_s = (current_k_global_token_idx_for_s >=
                                         current_kv_cache_total_len);
              bool causal_mask_applies =
                  (current_k_global_token_idx_for_s > q_global_pos_for_causal);

              if (q_is_padding_for_s || k_is_padding_for_s ||
                  causal_mask_applies) {
                s_tmp[r_smem_q_idx * B_c + c_smem_k_idx] = -FLT_MAX;
              } else {
                s_tmp[r_smem_q_idx * B_c + c_smem_k_idx] = final_s_cell_val;
              }
            }
          }
        }
      }
      __syncthreads();  // S_ij计算完毕, s_tmp对所有线程可见

      // --- 在线Softmax 和 输出累积 (O_i += P_ij * V_j) ---
      for (int q_row_offset_in_warp_softmax = 0;
           q_row_offset_in_warp_softmax < NUM_Q_ROWS_PER_WARP;
           ++q_row_offset_in_warp_softmax) {
        int r_smem_idx = warp_idx_in_block * NUM_Q_ROWS_PER_WARP +
                         q_row_offset_in_warp_softmax;

        if (r_smem_idx >= B_r) continue;

        int q_token_relative_idx_in_prefill_softmax =
            q_segment_global_start_token_idx + i_q_chunk_offset_in_segment +
            r_smem_idx;
        if (q_token_relative_idx_in_prefill_softmax >= current_prefill_q_length)
          continue;

        float m_prev_row = m_stats[r_smem_idx];
        float l_prev_row = l_stats[r_smem_idx];

        float m_block_curr = -FLT_MAX;
        for (int c_col_max = lane_idx_in_warp; c_col_max < B_c;
             c_col_max += warpSize) {
          m_block_curr = max(m_block_curr, s_tmp[r_smem_idx * B_c + c_col_max]);
        }
        m_block_curr = warpReduceMax(
            m_block_curr);  // 结果已广播，所有lane拿到正确的m_block_curr

        float m_new_row = max(m_prev_row, m_block_curr);
        float scale_factor = expf(m_prev_row - m_new_row);

        // lane 0 更新 l_stats (部分)
        if (lane_idx_in_warp == 0) {
          l_stats[r_smem_idx] = l_prev_row * scale_factor;
        }
        // 所有线程并行缩放 o_smem 对应行
        for (int d_col_scale = lane_idx_in_warp; d_col_scale < DQKV;
             d_col_scale += warpSize) {
          o_smem[r_smem_idx * DQKV + d_col_scale] = static_cast<T>(
              static_cast<float>(o_smem[r_smem_idx * DQKV + d_col_scale]) *
              scale_factor);
        }

        float l_block_curr_partial = 0.0f;
        for (int c_col_p = lane_idx_in_warp; c_col_p < B_c;
             c_col_p += warpSize) {
          float s_val = s_tmp[r_smem_idx * B_c + c_col_p];
          float p_val = (s_val == -FLT_MAX) ? 0.0f : expf(s_val - m_new_row);
          s_tmp[r_smem_idx * B_c + c_col_p] = p_val;  // P_ij存回s_tmp
          l_block_curr_partial += p_val;
        }
        float l_block_curr_sum =
            warpReduceSum(l_block_curr_partial);  // 结果已广播

        if (lane_idx_in_warp == 0) {
          l_stats[r_smem_idx] +=
              l_block_curr_sum;             // lane 0 更新 l_stats (另一部分)
          m_stats[r_smem_idx] = m_new_row;  // lane 0 更新 m_stats
        }
        __syncthreads();  // 确保m_stats,
                          // l_stats更新，P_ij在s_tmp中对所有线程可见

        for (int d_col_out = lane_idx_in_warp; d_col_out < DQKV;
             d_col_out += warpSize) {
          float sum_pv = 0.0f;
          for (int c_col_v = 0; c_col_v < B_c; ++c_col_v) {
            sum_pv += s_tmp[r_smem_idx * B_c + c_col_v] *
                      static_cast<float>(v_smem[c_col_v * DQKV + d_col_out]);
          }
          o_smem[r_smem_idx * DQKV + d_col_out] = static_cast<T>(
              static_cast<float>(o_smem[r_smem_idx * DQKV + d_col_out]) +
              sum_pv);
        }
      }  // 结束对Warp内Q行的Softmax和输出累积处理
      __syncthreads();  // 当前K_j, V_j块处理完毕，同步所有Warp
    }  // 结束内层K/V块循环

    constexpr int NUM_Q_ROWS_PER_WARP_WRITE = (B_r + WARP_NUM - 1) / WARP_NUM;
    for (int q_row_offset_in_warp_write = 0;
         q_row_offset_in_warp_write < NUM_Q_ROWS_PER_WARP_WRITE;
         ++q_row_offset_in_warp_write) {
      int r_smem_idx_write = warp_idx_in_block * NUM_Q_ROWS_PER_WARP_WRITE +
                             q_row_offset_in_warp_write;

      if (r_smem_idx_write >= B_r) continue;

      int current_q_token_relative_idx_in_prefill_write =
          q_segment_global_start_token_idx + i_q_chunk_offset_in_segment +
          r_smem_idx_write;

      if (current_q_token_relative_idx_in_prefill_write <
          current_prefill_q_length) {
        float l_final_val = l_stats[r_smem_idx_write];
        float l_final_inv = 1.0f / l_final_val;

        T* out_global_row_ptr =
            out_global +
            (current_q_token_relative_idx_in_prefill_write * num_q_heads_total +
             q_head_idx_global) *
                DQKV;

        for (int d_col_write = lane_idx_in_warp; d_col_write < DQKV;
             d_col_write += warpSize) {
          out_global_row_ptr[d_col_write] = static_cast<T>(
              static_cast<float>(
                  o_smem[r_smem_idx_write * DQKV + d_col_write]) *
              l_final_inv);
        }
      }
    }
    __syncthreads();  // Q_i块输出完毕，同步
  }  // 结束外层Q_i块循环
}

// Host函数：Flash Attention Prefill的tensor接口
// (Host函数部分保持和你原来的一致，仅修改Kernel调用处的共享内存变量名)
template <typename T>
void flash_attention_prefill(
    const Tensor<T>& Q,  // Query张量 [seq_len, n_heads, head_dim]
    const Tensor<T>& K,  // Key张量 [total_seq_len, n_kv_heads, head_dim]
    const Tensor<T>& V,  // Value张量 [total_seq_len, n_kv_heads, head_dim]
    Tensor<T>& output,   // 输出张量 [seq_len, n_heads, head_dim]
    int n_heads,         // Query头数
    int n_kv_heads,      // Key/Value头数
    int head_dim,        // 头维度
    int seq_len,         // Query序列长度
    int total_seq_len,   // Key/Value总序列长度 (这个参数在kernel中对应
                         // current_kv_cache_total_len)
    int offset,          // Q在整个序列中的起始偏移量
    cudaStream_t stream) {
  if (head_dim != 128) {
    throw std::runtime_error(
        "Flash attention prefill currently only supports head_dim=128");
  }

  int n_groups = n_heads / n_kv_heads;

  // 模板参数设置 - 和你原来保持一致
  constexpr int B_c = 32;
  constexpr int B_r = 8;
  constexpr int NUM_ELEM_PER_THREAD = 8;  // Kernel中未使用，但保留
  constexpr int WARP_NUM = 2;
  constexpr int DQKV_val = 128;  // 使用不同的名字避免和kernel内DQKV冲突

  // 计算动态共享内存大小 (和Kernel内声明顺序、类型、大小匹配)
  size_t q_smem_bytes = B_r * DQKV_val * sizeof(T);
  size_t k_smem_bytes = B_c * DQKV_val * sizeof(T);
  size_t v_smem_bytes = B_c * DQKV_val * sizeof(T);
  size_t s_tmp_bytes = B_r * B_c * sizeof(float);
  size_t o_smem_bytes = B_r * DQKV_val * sizeof(T);
  size_t m_stats_bytes = B_r * sizeof(float);
  size_t l_stats_bytes = B_r * sizeof(float);

  size_t shared_mem_size = q_smem_bytes + k_smem_bytes + v_smem_bytes +
                           s_tmp_bytes + o_smem_bytes + m_stats_bytes +
                           l_stats_bytes;

  // 确保共享内存大小至少为1字节，以防所有参数为0的情况（虽然不太可能）
  if (shared_mem_size == 0) shared_mem_size = 1;

  // 根据序列长度选择合适的模板参数
  if (seq_len <= 512) {
    constexpr int T_r = 512;
    constexpr int T_c_val = 2048;  // T_c 对应 kernel 内的 K/V cache
                                   // 上限，实际迭代由 total_seq_len 控制
    constexpr int divide_len = 512;

    int num_q_segments = (seq_len + divide_len - 1) / divide_len;
    dim3 grid(num_q_segments, n_heads);
    dim3 block(WARP_NUM * 32);  // 假设 warpSize = 32

    flash_attn_prefill_kernel<T, B_c, B_r, T_r, T_c_val, divide_len,
                              NUM_ELEM_PER_THREAD, WARP_NUM, DQKV_val>
        <<<grid, block, shared_mem_size, stream>>>(
            Q.data_ptr(),       // q_global
            K.data_ptr(),       // k_global
            V.data_ptr(),       // v_global
            output.data_ptr(),  // out_global
            n_heads,            // num_q_heads_total
            n_kv_heads,         // num_kv_heads_total
            n_groups,           // GQA_n_group
            seq_len,            // current_prefill_q_length
            total_seq_len,      // current_kv_cache_total_len
            offset              // q_offset_in_kv_timeline
        );
  } else {
    constexpr int T_r = 2048;
    constexpr int T_c_val = 2048;
    constexpr int divide_len = 2048;

    int num_q_segments = (seq_len + divide_len - 1) / divide_len;
    dim3 grid(num_q_segments, n_heads);
    dim3 block(WARP_NUM * 32);

    flash_attn_prefill_kernel<T, B_c, B_r, T_r, T_c_val, divide_len,
                              NUM_ELEM_PER_THREAD, WARP_NUM, DQKV_val>
        <<<grid, block, shared_mem_size, stream>>>(
            Q.data_ptr(), K.data_ptr(), V.data_ptr(), output.data_ptr(),
            n_heads, n_kv_heads, n_groups, seq_len, total_seq_len, offset);
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error("CUDA error in flash_attention_prefill: " +
                             std::string(cudaGetErrorString(err)));
  }
}

// 显式实例化 (保持和你原来的一致)
template void flash_attention_prefill<float>(
    const Tensor<float>& Q, const Tensor<float>& K, const Tensor<float>& V,
    Tensor<float>& output, int n_heads, int n_kv_heads, int head_dim,
    int seq_len, int total_seq_len, int offset, cudaStream_t stream);

template void flash_attention_prefill<__nv_bfloat16>(
    const Tensor<__nv_bfloat16>& Q, const Tensor<__nv_bfloat16>& K,
    const Tensor<__nv_bfloat16>& V, Tensor<__nv_bfloat16>& output, int n_heads,
    int n_kv_heads, int head_dim, int seq_len, int total_seq_len, int offset,
    cudaStream_t stream);

#ifdef ENABLE_FP16_TYPES  // Guard for __half if not always available/included
template void flash_attention_prefill<__half>(
    const Tensor<__half>& Q, const Tensor<__half>& K, const Tensor<__half>& V,
    Tensor<__half>& output, int n_heads, int n_kv_heads, int head_dim,
    int seq_len, int total_seq_len, int offset, cudaStream_t stream);
#endif

}  // namespace cuda_OP