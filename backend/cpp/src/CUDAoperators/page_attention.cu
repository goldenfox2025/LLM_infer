// /*
//  * 版权声明和许可信息 (源自 vLLM 和 FasterTransformer)
//  * Copyright (c) 2023, The vLLM team.
//  * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
//  *
//  * Licensed under the Apache License, Version 2.0 (the "License");
//  * you may not use this file except in compliance with the License.
//  * You may obtain a copy of the License at
//  *
//  * http://www.apache.org/licenses/LICENSE-2.0
//  *
//  * Unless required by applicable law or agreed to in writing, software
//  * distributed under the License is distributed on an "AS IS" BASIS,
//  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  * See the License for the specific language governing permissions and
//  * limitations under the License.
//  */

// #ifndef VLLM_PAGED_ATTENTION_KERNELS_HPP
// #define VLLM_PAGED_ATTENTION_KERNELS_HPP

// // 基础 CUDA 和 C++ 头文件
// #include <cuda_bf16.h>  // __nv_bfloat16 数据类型及相关转换函数
// #include <cuda_runtime.h>  // CUDA 运行时 API, 设备函数如 __shfl_*, __expf,
// __fdividef 等

// #include <cassert>  // 包含 assert 宏，用于调试时的断言检查
// #include <cfloat>   // 包含 FLT_MAX 等浮点数相关的宏

// // 定义 WARP_SIZE，在 NVIDIA GPU 上通常是 32 个线程组成一个 Warp
// #define WARP_SIZE 32

// // 常用宏定义
// #define MAX(a, b) ((a) > (b) ? (a) : (b))              //
// 返回两个值中的最大值 #define MIN(a, b) ((a) < (b) ? (a) : (b)) //
// 返回两个值中的最小值 #define DIVIDE_ROUND_UP(a, b) (((a) + (b) - 1) / (b)) //
// 执行向上取整的除法

// namespace vllm {

// // --- 类型转换和零值工具函数 ---
// // 这些函数用于在不同数值类型（特别是 __nv_bfloat16 和 float）之间进行转换，
// // 以及将变量初始化为零。

// /**
//  * @brief 将变量设置为其类型的零值。
//  * @tparam T 变量的类型。
//  * @param val 要设置为零的变量的引用。
//  */
// template <typename T>
// __device__ inline void zero(T& val) {
//   val = static_cast<T>(0);
// }

// /**
//  * @brief __nv_bfloat16 类型的零值特化。
//  * @param val 要设置为零的 __nv_bfloat16 变量的引用。
//  */
// template <>
// __device__ inline void zero(__nv_bfloat16& val) {
//   val = __float2bfloat16(
//       0.0f);  // 使用 CUDA 内建函数将 float 0.0f 转换为 bfloat16
// }

// /**
//  * @brief 将任意类型 T 的值转换为 float 类型。
//  * @tparam T 输入值的类型。
//  * @param val 要转换的值。
//  * @return 转换后的 float 值。
//  */
// template <typename T>
// __device__ inline float to_float(T val) {
//   return static_cast<float>(val);
// }

// /**
//  * @brief __nv_bfloat16 类型转换为 float 的特化。
//  * @param val 要转换的 __nv_bfloat16 值。
//  * @return 转换后的 float 值。
//  */
// template <>
// __device__ inline float to_float(__nv_bfloat16 val) {
//   return __bfloat162float(val);  // 使用 CUDA 内建函数将 bfloat16 转换为
//   float
// }

// /**
//  * @brief 将 float 值转换为任意类型 T，并赋值给变量。
//  * @tparam T 目标变量的类型。
//  * @param var 目标变量的引用。
//  * @param val 要转换的 float 值。
//  */
// template <typename T>
// __device__ inline void from_float(T& var, float val) {
//   var = static_cast<T>(val);
// }

// /**
//  * @brief float 值转换为 __nv_bfloat16 类型的特化。
//  * @param var 目标 __nv_bfloat16 变量的引用。
//  * @param val 要转换的 float 值。
//  */
// template <>
// __device__ inline void from_float(__nv_bfloat16& var, float val) {
//   var = __float2bfloat16(val);  // 使用 CUDA 内建函数将 float 转换为 bfloat16
// }

// // --- 向量化数据类型辅助结构体 ---

// /**
//  * @brief 模拟一个固定大小的向量类型。
//  * 用于在 CUDA 核函数中进行向量化的数据加载和计算。
//  * @tparam T 向量元素的数据类型。
//  * @tparam N 向量中的元素数量。
//  */
// template <typename T, int N>
// struct VecType {
//   T data[N];  // 存储向量数据的数组

//   // 通过 [] 操作符访问向量元素
//   __device__ T& operator[](int i) { return data[i]; }
//   __device__ const T& operator[](int i) const { return data[i]; }

//   // 默认构造函数
//   __device__ VecType() {}

//   // 从指针构造向量 (简化示例，用于从内存加载)
//   __device__ VecType(const T* ptr) {
//     for (int i = 0; i < N; ++i) data[i] = ptr[i];
//   }
// };

// // --- 向量类型转换工具函数 ---

// /**
//  * @brief 将一个浮点类型的向量 (VecType<float, N>) 转换为目标标量类型的向量
//  * (VecType<DestScalarT, N>)。
//  * @tparam DestScalarT 目标向量元素的数据类型。
//  * @tparam SrcFloatVecT 源浮点向量的类型 (通常是 VecType<float, N>)。
//  * @tparam N 向量中的元素数量。
//  * @param dest 转换结果的目标向量。
//  * @param src 要转换的源浮点向量。
//  */
// template <typename DestScalarT, typename SrcFloatVecT, int N>
// __device__ inline void convert_float_vec_to_scalar_vec(
//     VecType<DestScalarT, N>& dest, const SrcFloatVecT& src) {
//   // SrcFloatVecT 应该是 VecType<float, N> 或类似的结构
//   for (int i = 0; i < N; ++i) {
//     from_float(dest.data[i], src.data[i]);  // 使用 from_float 进行逐元素转换
//   }
// }

// // --- QK 点积计算结构体 ---

// /**
//  * @brief QK_dot 结构体封装了查询向量 (Q) 和键向量 (K) 之间点积的计算逻辑。
//  * 这个版本的 `dot` 方法在其内部包含了跨 THREAD_GROUP_SIZE 个线程的归约操作。
//  *
//  * @tparam scalar_t Q, K, V 和中间计算所使用的数据类型 (例如 float,
//  * __nv_bfloat16)。
//  * @tparam THREAD_GROUP_SIZE 参与一次点积计算并进行归约的线程组的大小。
//  * 这些线程应位于同一个 CUDA Warp 内，且 THREAD_GROUP_SIZE 通常是2的幂。
//  * 这个写法应该是为了以后扩展
//  * 模板元编程我这辈子都看不懂了
//  */
// template <typename scalar_t, int THREAD_GROUP_SIZE>
// struct Qk_dot {
//   /**
//    * @brief 计算 Q 和 K 向量部分的点积，并在 THREAD_GROUP_SIZE
//    * 个线程间进行归约。
//    *
//    * @tparam Q_vec_array_type 当前线程持有的 Q 向量部分数组的类型 (例如
//    * VecType<scalar_t, VEC_SIZE_PARAM>[NUM_VECS_PER_THREAD])。
//    * @tparam K_vec_array_type 当前线程持有的 K 向量部分数组的类型。
//    * @tparam NUM_VECS_PER_THREAD 每个线程负责处理的 Q/K 子向量的数量
//    * (用于构成完整的 head_size)。
//    * @tparam VEC_SIZE_PARAM 每个子向量的元素数量。
//    *
//    * @param q_vecs_for_one_thread 当前线程负责的 Q 向量部分。
//    * @param k_vecs_for_one_token 当前线程负责的 K 向量部分 (对应一个 K
//    token)。
//    * @return float 返回经过归约后的 QK 点积总和。组内的所有 THREAD_GROUP_SIZE
//    * 个线程都会得到相同的结果。
//    * 可以安排多个线程负责这个向量乘
//    */
//   template <typename Q_vec_array_type, typename K_vec_array_type,
//             int NUM_VECS_PER_THREAD, int VEC_SIZE_PARAM>
//   __device__ static float dot(const Q_vec_array_type& q_vecs_for_one_thread,
//                               const K_vec_array_type& k_vecs_for_one_token) {
//     float qk_val = 0.0f;  // 每个线程计算的局部点积和

// // 步骤 1: 计算当前线程负责的 Q、K 向量部分的点积和
// // 遍历每个线程负责的子向量块
// #pragma unroll
//     for (int i = 0; i < NUM_VECS_PER_THREAD; ++i) {
// // 遍历子向量块内的每个元素 当然是用fp32保持精度
// #pragma unroll
//       for (int k = 0; k < VEC_SIZE_PARAM; ++k) {
//         qk_val += to_float(q_vecs_for_one_thread[i].data[k]) *
//                   to_float(k_vecs_for_one_token[i].data[k]);
//       }
//     }

//     // 步骤 2: 在 THREAD_GROUP_SIZE 个线程间进行归约 (求和)
//     // 使用 __shfl_xor_sync 指令在 Warp 内的线程间高效交换数据并累加。
//     // 此方法要求 THREAD_GROUP_SIZE 是 2 的幂，并且不大于 WARP_SIZE。
//     if (THREAD_GROUP_SIZE > 1) {
// #pragma unroll
//       for (int offset = THREAD_GROUP_SIZE / 2; offset > 0; offset /= 2) {
//         // __shfl_xor_sync(mask, var, laneMask, width)
//         // - mask: 一个32位整数，通常为 0xFFFFFFFF，表示 Warp
//         // 内所有激活线程参与同步和 shuffle 操作。
//         // - var: 要进行 shuffle 的变量 (即 qk_val)。
//         // - laneMask: 一个整数 (这里是 offset)，与调用线程的 lane ID (线程在
//         // Warp 内的索引) 进行异或 (XOR) 操作，
//         //             得到源 lane 的 ID。数据将从这个源 lane 读取。
//         // - width: shuffle 操作的逻辑宽度，这里是 THREAD_GROUP_SIZE。线程的
//         // lane ID 会被隐式地对 width 取模，
//         //          确保 shuffle 操作在指定的 THREAD_GROUP_SIZE
//         线程子集内进行。
//         // 举个例子，每个线程group大小是4
//         // 32 / 4 = 8
//         // 比如调用 lane id 是 1
//         // 第一次，offset是2
//         // 01和10进行xor
//         // 返回11
//         // 1会从3获取数据
//         // 3会从1获取数据
//         // 0 2
//         // 2 0
//         // 再来一次 01 23
//         // 整个group内部就都归约到了最大值
//         qk_val +=
//             __shfl_xor_sync(0xFFFFFFFF, qk_val, offset, THREAD_GROUP_SIZE);
//       }
//     }
//     // 归约完成后，qk_val 在 THREAD_GROUP_SIZE
//     // 范围内的所有参与线程中都持有最终的、相同的总和。
//     return qk_val;
//   }
// };

// // --- 通用向量点积函数 ---

// /**
//  * @brief 计算两个给定类型和长度的向量的点积。
//  * 主要用于计算 (softmax probabilities P_ij) 和 (Value 向量 V_j) 的点积。
//  * @tparam LhsScalarT 左侧向量元素的数据类型。
//  * @tparam RhsScalarT 右侧向量元素的数据类型。
//  * @tparam N 向量的长度。
//  * @param lhs 左侧向量。
//  * @param rhs 右侧向量。
//  * @return float 返回两个向量的点积结果。
//  */
// template <typename LhsScalarT, typename RhsScalarT, int N>
// __device__ inline float dot_vectors(const VecType<LhsScalarT, N>& lhs,
//                                     const VecType<RhsScalarT, N>& rhs) {
//   float sum = 0.0f;
// #pragma unroll
//   for (int i = 0; i < N; ++i) {
//     sum += to_float(lhs.data[i]) * to_float(rhs.data[i]);
//   }
//   return sum;
// }

// // --- Attention Softmax 辅助函数：块内求和 ---

// /**
//  * @brief 在一个 CUDA 线程块 (Block) 内，对所有线程持有的浮点值进行求和。
//  * 使用共享内存和 Warp Shuffle 指令进行高效的并行归约。
//  *
//  * @tparam NUM_WARPS_PARAM 线程块中参与计算的 Warp 数量。
//  * @param red_smem 指向共享内存中用于归约的工作区 (大小至少为 NUM_WARPS_PARAM
//  *
//  * sizeof(float))。
//  * @param sum 当前线程持有的、需要被累加到总和中的浮点值。
//  * @return float 返回整个线程块的总和。该总和会被广播给块内所有线程。
//  */
// #define VLLM_SHFL_XOR_SYNC(var, lane_mask) \
//   __shfl_xor_sync(uint32_t(-1), var, lane_mask)

// #define VLLM_SHFL_SYNC(var, src_lane) __shfl_sync(uint32_t(-1), var,
// src_lane) template <int NUM_WARPS> inline __device__ float block_sum(float*
// red_smem, float sum) {
//   // 索引
//   int warp = threadIdx.x / WARP_SIZE;
//   int lane = threadIdx.x % WARP_SIZE;

//   // 计算每个warp的总和
// #pragma unroll
//   for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
//     sum += VLLM_SHFL_XOR_SYNC(sum, mask);
//   }

//   // warp存入共享内存
//   if (lane == 0) {
//     red_smem[warp] = sum;
//   }
//   __syncthreads();

//   // 最后，一个warp做归约
//   if (lane < NUM_WARPS) {
//     sum = red_smem[lane];
//   }
// //
// #pragma unroll
//   for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
//     sum += VLLM_SHFL_XOR_SYNC(sum, mask);
//   }

//   return VLLM_SHFL_SYNC(sum, 0);
// }

// // --- Paged Attention CUDA 核函数 (核心设备逻辑) ---

// /**
//  * @brief Paged Attention 的核心 CUDA 设备端计算逻辑。
//  * 此函数被 `paged_attention_v1_kernel` 和 `paged_attention_v2_kernel` 调用。
//  * 它处理单个查询向量与一系列分页存储的键/值向量之间的注意力计算。
//  *
//  * CUDA Kernel 启动配置 (典型):
//  * - Grid: (num_attention_heads, num_sequences,
//  * num_partitions_for_v2_or_1_for_v1)
//  * - Block: (NUM_THREADS)
//  *
//  * @tparam scalar_t Q, O (输出) 以及中间计算使用的数据类型 (例如 float,
//  * __nv_bfloat16)。
//  * @tparam cache_t K, V Cache 中存储的数据类型 (例如 __nv_bfloat16)。
//  * @tparam HEAD_SIZE 每个注意力头的维度大小。
//  * @tparam BLOCK_SIZE KV Cache 中每个物理 block 存储的 token 数量。
//  * @tparam NUM_THREADS 每个 CUDA 线程块中的线程数量。
//  * @tparam IS_BLOCK_SPARSE 是否启用块稀疏注意力机制 (布尔值)。
//  * @tparam PARTITION_SIZE (仅用于 V2 版本) 序列分区的大小。如果为
//  * 0，表示不进行分区 (V1 版本)。
//  *
//  * @param exp_sums (仅 V2) 输出参数，存储 softmax 分母的部分和 (exp_sum)。
//  * 维度: [num_seqs, num_heads, max_num_partitions]。
//  * @param max_logits (仅 V2) 输出参数，存储每个分区内 QK 点积的最大值
//  * (max_logit)。 维度: [num_seqs, num_heads, max_num_partitions]。
//  * @param out 输出参数，存储注意力计算的结果。
//  * V1 维度: [num_seqs, num_heads, head_size]。
//  * V2 维度 (tmp_out): [num_seqs, num_heads, max_num_partitions, head_size]。
//  * @param q 输入参数，查询向量 (Query)。维度: [num_seqs, num_heads,
//  head_size]。
//  * @param k_cache 输入参数，分页存储的键缓存 (Key Cache)。
//  * 维度: [num_physical_blocks, num_kv_heads, head_size/X, BLOCK_SIZE, X] (X
//  * 是打包因子)。
//  * @param v_cache 输入参数，分页存储的值缓存 (Value Cache)。
//  * 维度: [num_physical_blocks, num_kv_heads, head_size, BLOCK_SIZE]。
//  * @param num_kv_heads KV 头的数量 (用于支持 Grouped-Query Attention)。
//  * @param scale QK 点积的缩放因子 (通常是 1.0f / sqrt(HEAD_SIZE))。
//  * @param block_tables 输入参数，Block 表，将逻辑 token
//  位置映射到物理存储块。
//  * 维度: [num_seqs, max_num_blocks_per_seq]。
//  * @param seq_lens 输入参数，每个序列的实际长度。维度: [num_seqs]。
//  * @param max_num_blocks_per_seq 每个序列可能占用的最大 block 数量。
//  * @param alibi_slopes 输入参数，ALiBi 位置偏置的斜率。维度:
//  [num_heads]。如果为
//  * nullptr，则禁用 ALiBi。
//  * @param q_stride Q 张量在序列维度上的步长 (通常是 num_heads * head_size)。
//  * @param kv_block_stride K/V Cache 中每个物理 block 的步长。
//  * @param kv_head_stride K/V Cache 中每个 KV 头的步长。
//  * @param tp_rank (用于块稀疏) Tensor Parallelism 的 rank。
//  * @param blocksparse_local_blocks (用于块稀疏) 本地关注的块数量。
//  * @param blocksparse_vert_stride (用于块稀疏) 垂直步长。
//  * @param blocksparse_block_size (用于块稀疏) 稀疏模式下的块大小。
//  * @param blocksparse_head_sliding_step (用于块稀疏) 头滑动的步长。
//  */
// template <typename scalar_t, typename cache_t, int HEAD_SIZE, int BLOCK_SIZE,
//           int NUM_THREADS, bool IS_BLOCK_SPARSE, int PARTITION_SIZE = 0>
// __device__ void paged_attention_kernel(
//     float* __restrict__ exp_sums, float* __restrict__ max_logits,
//     scalar_t* __restrict__ out, const scalar_t* __restrict__ q,
//     const cache_t* __restrict__ k_cache, const cache_t* v_cache,
//     const int num_kv_heads, const float scale,
//     const int* __restrict__ block_tables, const int* __restrict__ seq_lens,
//     const int max_num_blocks_per_seq, const float* __restrict__ alibi_slopes,
//     const int q_stride, const int kv_block_stride, const int kv_head_stride,
//     const int tp_rank, const int blocksparse_local_blocks,
//     const int blocksparse_vert_stride, const int blocksparse_block_size,
//     const int blocksparse_head_sliding_step) {
//   // --- 1. 计算线程和块索引，确定处理范围 ---
//   const int seq_idx = blockIdx.y;  // 当前线程块处理的序列 (sequence) 索引。
//   const int partition_idx =
//       blockIdx.z;  // 当前线程块处理的分区 (partition) 索引 (仅 V2
//       版本有意义)。
//   const int max_num_partitions = gridDim.z;  // 总的分区数 (仅 V2
//   版本有意义)。

//   constexpr bool USE_PARTITIONING =
//       (PARTITION_SIZE > 0);               // 判断是否为 V2 分区模式。
//   const int seq_len = seq_lens[seq_idx];  // 获取当前序列的实际长度。

//   // 如果启用了分区
//   //
//   (V2)，并且当前分区索引超出了序列的有效范围，则该线程块无需工作，提前返回。
//   if (USE_PARTITIONING && (partition_idx * PARTITION_SIZE >= seq_len)) {
//     return;
//   }

//   // 计算当前序列总共包含多少个逻辑 KV Block。
//   const int num_seq_blocks = DIVIDE_ROUND_UP(seq_len, BLOCK_SIZE);
//   // 计算当前分区包含多少个 KV Block。如果是非分区模式 (V1)，则等于整个序列的
//   // Block 数。
//   const int num_blocks_per_partition =
//       USE_PARTITIONING ? (PARTITION_SIZE / BLOCK_SIZE) : num_seq_blocks;

//   // 确定当前线程块需要处理的 KV Block 的逻辑索引范围 [start_block_idx,
//   // end_block_idx)。
//   const int start_block_idx =  // 起始逻辑 Block 索引。
//       USE_PARTITIONING ? (partition_idx * num_blocks_per_partition) : 0;
//   const int end_block_idx =  // 结束逻辑 Block 索引 (不包含)。
//       MIN(start_block_idx + num_blocks_per_partition, num_seq_blocks);
//   // const int num_blocks_to_process = end_block_idx - start_block_idx; //
//   // 当前分区/序列中实际要处理的Block数量

//   // 确定当前线程块需要处理的 Token 的逻辑索引范围 [start_token_idx,
//   // end_token_idx)。
//   const int start_token_idx =
//       start_block_idx * BLOCK_SIZE;  // 起始 Token 逻辑索引。
//   const int end_token_idx =          // 结束 Token 逻辑索引 (不包含)。
//       MIN(start_token_idx + (end_block_idx - start_block_idx) * BLOCK_SIZE,
//           seq_len);
//   const int num_tokens_in_partition =
//       end_token_idx -
//       start_token_idx;  // 当前分区/序列中实际要处理的 Token 数量。

//   // --- 2. 配置线程组和向量化参数 ---
//   // THREAD_GROUP_SIZE: 一个 Warp 内，多少个线程协作处理一个 K/V token 的
//   // head_size 维度。 目的是为了向量化加载 K/V 数据并进行点积计算。
//   constexpr int THREAD_GROUP_SIZE = MAX(WARP_SIZE / BLOCK_SIZE, 1);
//   // NUM_THREAD_GROUPS: 线程块内有多少个这样的 THREAD_GROUP。
//   constexpr int NUM_THREAD_GROUPS_IN_BLOCK = NUM_THREADS / THREAD_GROUP_SIZE;
//   assert(NUM_THREADS % THREAD_GROUP_SIZE == 0);  // 确保可以整除

//   // NUM_TOKENS_PER_ITER_PER_WARP_LANE_GROUP: 一个 Warp Lane
//   // (可以看作一个线程组，如果 THREAD_GROUP_SIZE=1 则是一个线程)
//   // 在内层循环的一次迭代中处理的 K token 数量。这通常用于将 BLOCK_SIZE 内的
//   // tokens 分配给 Warp 内的 Lanes。
//   constexpr int NUM_K_TOKENS_PER_BLOCK_ITER =
//       DIVIDE_ROUND_UP(BLOCK_SIZE, WARP_SIZE);

//   constexpr int NUM_WARPS_IN_BLOCK =
//       NUM_THREADS / WARP_SIZE;  // 线程块中的 Warp 数量。
//   const int thread_global_idx =
//       threadIdx.x;  // 线程在块内的全局索引 (0 到 NUM_THREADS-1)。
//   const int warp_global_idx =
//       thread_global_idx / WARP_SIZE;  // 线程所在的 Warp 在块内的索引。
//   const int lane_idx =
//       thread_global_idx %
//       WARP_SIZE;  // 线程在其 Warp 内的索引 (Lane ID, 0 到 WARP_SIZE-1)。

//   // 计算注意力头相关索引。
//   const int head_idx = blockIdx.x;    // 当前线程块处理的注意力头 (Q头)
//   索引。 const int num_q_heads = gridDim.x;  // 总的查询头 (Q头) 数量。 const
//   int num_q_per_kv_head =
//       num_q_heads / num_kv_heads;  // GQA: 每个KV头对应多少个Q头。
//   const int kv_head_idx =
//       head_idx / num_q_per_kv_head;  // 当前Q头对应的KV头索引。

//   // 获取 ALiBi 斜率 (如果启用)。
//   const float alibi_slope =
//       (alibi_slopes == nullptr) ? 0.f : alibi_slopes[head_idx];

//   // 向量化加载/存储的配置。目标是每次内存操作处理约16字节以提高效率。
//   // VEC_SIZE_Q: 加载查询向量 Q 时，每个线程一次处理的 scalar_t 元素数量。
//   constexpr int VEC_SIZE_Q =
//       MAX(1, 16 / (THREAD_GROUP_SIZE * sizeof(scalar_t)));
//   using Q_vec = VecType<scalar_t, VEC_SIZE_Q>;  // Q 子向量类型。

//   // VEC_SIZE_K_CACHE: 加载键缓存 K 时，每个线程一次处理的 cache_t 元素数量。
//   constexpr int VEC_SIZE_K_CACHE =
//       MAX(1, 16 / (THREAD_GROUP_SIZE * sizeof(cache_t)));
//   // K_vec: 用于存储从 K Cache (cache_t 类型) 转换到计算类型 (scalar_t) 后的
//   K
//   // 子向量。 其元素数量应与 Q_vec 匹配以便点积。
//   using K_vec = VecType<scalar_t, VEC_SIZE_Q>;
//   using K_cache_vec =
//       VecType<cache_t, VEC_SIZE_K_CACHE>;  // K Cache 子向量类型。

//   // NUM_ELEMS_PER_THREAD_IN_HEAD: 在 head_size
//   维度上，每个线程负责的元素数量。 constexpr int NUM_ELEMS_PER_THREAD_IN_HEAD
//   = HEAD_SIZE / THREAD_GROUP_SIZE;
//   // NUM_Q_VECS_PER_THREAD: 每个线程负责的 Q 子向量的数量。
//   constexpr int NUM_Q_VECS_PER_THREAD =
//       NUM_ELEMS_PER_THREAD_IN_HEAD / VEC_SIZE_Q;
//   assert(HEAD_SIZE % THREAD_GROUP_SIZE ==
//          0);  // 确保 head_size 能被线程组大小整除。
//   assert(NUM_ELEMS_PER_THREAD_IN_HEAD % VEC_SIZE_Q ==
//          0);  // 确保每个线程的元素数能被向量大小整除。

//   // 线程在 THREAD_GROUP_SIZE 大小的“计算小组”内的索引。
//   const int thread_in_computation_group_offset =
//       thread_global_idx % THREAD_GROUP_SIZE;
//   // 线程所属的“计算小组”在整个线程块中的索引 (若以 THREAD_GROUP_SIZE
//   // 为单位划分)。
//   const int computation_group_idx_in_block =
//       thread_global_idx / THREAD_GROUP_SIZE;

//   // --- 3. 加载查询向量 Q 到共享内存 ---
//   // Q 向量的全局内存指针，指向当前 (seq_idx, head_idx) 的 Q 向量的起始位置。
//   const scalar_t* q_global_ptr = q + seq_idx * q_stride + head_idx *
//   HEAD_SIZE;

//   // 使用共享内存 __shared__ 来存储 Q 向量，以便线程组内的线程可以快速访问。
//   // 共享内存大小: THREAD_GROUP_SIZE * NUM_Q_VECS_PER_THREAD * VEC_SIZE_Q =
//   // HEAD_SIZE 个 scalar_t 元素。 q_vecs_shared[k][l] 表示第 k
//   // 个线程（在计算小组内）的第 l 个 Q 子向量。
//   __shared__ Q_vec q_vecs_shared[THREAD_GROUP_SIZE][NUM_Q_VECS_PER_THREAD];

// // 并行加载 Q 向量到共享内存。
// // 每个“计算小组” (computation_group_idx_in_block) 内的线程
// // (thread_in_computation_group_offset) 负责加载 Q 向量的不同部分。
// // computation_group_idx_in_block 遍历 NUM_Q_VECS_PER_THREAD，
// // thread_in_computation_group_offset 决定了加载 Q 向量的哪个“条带”。
// #pragma unroll
//   for (int i = computation_group_idx_in_block; i < NUM_Q_VECS_PER_THREAD;
//        i += NUM_THREAD_GROUPS_IN_BLOCK) {
//     // vec_elem_start_offset_in_q: 当前 Q 子向量在整个 Q 向量 (长度
//     HEAD_SIZE)
//     // 中的起始元素索引。
//     const int vec_elem_start_offset_in_q =
//         thread_in_computation_group_offset * VEC_SIZE_Q +  // 组内偏移
//         i * (THREAD_GROUP_SIZE * VEC_SIZE_Q);              // 组间大步进
//     if (vec_elem_start_offset_in_q < HEAD_SIZE) {          // 边界检查
//       q_vecs_shared[thread_in_computation_group_offset][i] =
//           *reinterpret_cast<const Q_vec*>(q_global_ptr +
//                                           vec_elem_start_offset_in_q);
//     }
//   }
//   __syncthreads();  // 同步，确保所有 Q 向量部分已从全局内存加载到共享内存。

//   // --- 4. 准备共享内存用于 Softmax 计算 ---
//   // `extern __shared__ char shared_mem_softmax_area[]`
//   // 表示使用动态分配的共享内存。 其总大小在核函数启动时由主机端指定。
//   extern __shared__ char shared_mem_softmax_area[];
//   // logits_smem: 用于存储 QK 点积结果 (logits)。使用 float32 以保证 Softmax
//   // 计算的精度。 大小: num_tokens_in_partition * sizeof(float)。
//   float* logits_smem = reinterpret_cast<float*>(shared_mem_softmax_area);
//   // reduction_smem: 用于后续归约操作 (如计算全局最大 logit 和全局 exp_sum)
//   // 的共享内存工作区。 需要 2 * NUM_WARPS_IN_BLOCK 个 float 空间。
//   float* reduction_smem = reinterpret_cast<float*>(
//       shared_mem_softmax_area + num_tokens_in_partition * sizeof(float));

//   // K Cache 的最后一个维度的大小，通常与向量化加载相关。
//   // 简化假设它等于 K Cache 向量加载的元素数量。
//   constexpr int X_DIM_K_CACHE_PACKING = VEC_SIZE_K_CACHE;

//   // 初始化当前线程计算的 QK 点积的局部最大值。
//   float qk_max_local_thread = -FLT_MAX;

//   // 获取当前序列在全局 block_tables 中的起始指针。
//   const int* block_table_for_seq =
//       block_tables + seq_idx * max_num_blocks_per_seq;

//   // --- 块稀疏注意力相关变量初始化 ---
//   int bs_sliding_offset = 0;        // 块稀疏：滑动窗口偏移量。
//   int q_sparse_block_id = 0;        // 块稀疏：查询向量 Q 所属的稀疏块 ID。
//   if constexpr (IS_BLOCK_SPARSE) {  // 仅在启用块稀疏时计算
//     q_sparse_block_id = (seq_len - 1) / blocksparse_block_size;
//     if (blocksparse_head_sliding_step >= 0) {  // Q 头滑动模式
//       bs_sliding_offset =
//           (tp_rank * num_q_heads + head_idx) * blocksparse_head_sliding_step
//           + 1;
//     } else {  // KV 头滑动模式
//       bs_sliding_offset = (tp_rank * num_kv_heads + kv_head_idx) *
//                               (-blocksparse_head_sliding_step) +
//                           1;
//     }
//   }

//   // --- 5. 迭代 Key blocks 计算 QK 点积 ---
//   // 每个 Warp (warp_global_idx) 负责处理当前分区内的一部分 KV Block。
//   // 使用步长 NUM_WARPS_IN_BLOCK 实现 Warp 间的负载均衡。
//   for (int current_logical_block_idx =
//            start_block_idx + warp_global_idx;      // Warp 的起始 Block
//        current_logical_block_idx < end_block_idx;  // 不超过分区的结束 Block
//        current_logical_block_idx += NUM_WARPS_IN_BLOCK) {  // 按 Warp
//        数量步进

//     // 从 Block 表中获取当前逻辑 Block 对应的物理存储块的编号。
//     int64_t physical_block_number =
//         static_cast<int64_t>(block_table_for_seq[current_logical_block_idx]);

//     // --- 块稀疏注意力: 判断是否跳过当前 Block ---
//     if constexpr (IS_BLOCK_SPARSE) {
//       const int k_sparse_block_id =
//           current_logical_block_idx * BLOCK_SIZE / blocksparse_block_size;
//       const bool is_remote_block =
//           ((k_sparse_block_id + bs_sliding_offset) % blocksparse_vert_stride
//           ==
//            0);
//       const bool is_local_block =
//           (k_sparse_block_id > q_sparse_block_id - blocksparse_local_blocks);

//       if (!is_remote_block &&
//           !is_local_block) {  // 如果既不是远程关注也不是本地关注，则跳过
//         // 对于跳过的 Token，将其在共享内存中的 logit 设置为极小值
//         (-FLT_MAX)。
//         // NUM_K_TOKENS_PER_BLOCK_ITER: 一个 Warp Lane 在内层循环中处理的 K
//         // Token 数量。 lane_idx 是 Warp 内的线程索引。
//         for (int k_token_iter = 0; k_token_iter <
//         NUM_K_TOKENS_PER_BLOCK_ITER;
//              ++k_token_iter) {
//           // 当前 Warp Lane (lane_idx) 负责的 Block 内的 K Token 偏移。
//           const int k_token_offset_in_block =
//               (lane_idx + k_token_iter * WARP_SIZE) % BLOCK_SIZE;
//           // 当前 K Token 在整个分区内的逻辑索引 (相对于分区的
//           // start_token_idx)。
//           const int k_token_idx_in_partition =
//               current_logical_block_idx * BLOCK_SIZE +
//               k_token_offset_in_block - start_token_idx;

//           if (k_token_idx_in_partition >= 0 &&
//               k_token_idx_in_partition < num_tokens_in_partition) {
//             // 只有“计算小组”内的第一个线程
//             (thread_in_computation_group_offset
//             // == 0) 写入共享内存。 这是因为 logits_smem 是按 token
//             索引的，一个
//             // token 对应一个 logit。
//             if (thread_in_computation_group_offset == 0) {
//               logits_smem[k_token_idx_in_partition] = -FLT_MAX;
//             }
//           }
//         }
//         continue;  // 跳过这个 Block 的后续 QK 计算。
//       }
//     }

//     // --- 加载 Key 向量并计算 QK 点积 (内层循环) ---
//     // 一个 Warp 内的线程 (由 lane_idx 区分) 协作处理一个 BLOCK_SIZE 的 K
//     // token。 每个“计算小组” (THREAD_GROUP_SIZE 个线程，由
//     // thread_in_computation_group_offset 区分) 协作处理 K 向量的 head_size
//     // 维度。 NUM_K_TOKENS_PER_BLOCK_ITER: 一个 Warp Lane
//     在此内层循环中处理的 K
//     // Token 数量。
//     for (int k_token_iter = 0; k_token_iter < NUM_K_TOKENS_PER_BLOCK_ITER;
//          ++k_token_iter) {
//       // 当前 Warp Lane (lane_idx) 负责的 Block 内的 K Token 的偏移量。
//       const int k_token_offset_in_block =
//           (lane_idx + k_token_iter * WARP_SIZE) % BLOCK_SIZE;
//       // 当前 K Token 在整个序列中的绝对逻辑索引。
//       const int k_token_abs_logical_idx =
//           current_logical_block_idx * BLOCK_SIZE + k_token_offset_in_block;

//       // K 子向量的临时寄存器存储 (用于当前线程处理的部分)。
//       K_vec k_vecs_reg[NUM_Q_VECS_PER_THREAD];  // K_vec 是 VecType<scalar_t,
//                                                 // VEC_SIZE_Q>

// // 从 K Cache 加载 K 向量的各个部分到寄存器。
// #pragma unroll
//       for (int q_vec_idx = 0; q_vec_idx < NUM_Q_VECS_PER_THREAD; ++q_vec_idx)
//       {
//         // vec_elem_start_offset_in_k: 当前线程负责的 K 向量部分在 head_size
//         // 维度上的起始元素索引。
//         const int vec_elem_start_offset_in_k =
//             thread_in_computation_group_offset * VEC_SIZE_K_CACHE +
//             q_vec_idx * (THREAD_GROUP_SIZE * VEC_SIZE_K_CACHE);

//         // 计算 K Cache 中对应数据的指针。
//         // K Cache 布局通常是 [num_blocks, num_kv_heads, block_size,
//         head_size]
//         // 或其变体。 这里假设 K Cache 中 head_size 维度是连续的，或者通过
//         // X_DIM_K_CACHE_PACKING 进行了打包。 为简化，假设 K Cache 中一个
//         token
//         // 的 head_size 数据是连续存储的。
//         const cache_t* k_cache_ptr_typed =
//             k_cache + physical_block_number * kv_block_stride +  //
//             定位到物理块 kv_head_idx * kv_head_stride + // 定位到 KV 头
//             k_token_offset_in_block * HEAD_SIZE +  // 定位到块内的 Token
//             vec_elem_start_offset_in_k;  // 定位到 Token 内的 head_size 部分

//         // 从 K Cache 加载 cache_t 类型的向量。
//         auto k_cache_vec_loaded =
//             *reinterpret_cast<const K_cache_vec*>(k_cache_ptr_typed);

//         // 类型转换: 从 cache_t 向量 (K_cache_vec) 转换为 scalar_t 向量
//         // (k_vecs_reg)。 VEC_SIZE_K_CACHE (用于加载K) 可能不等于 VEC_SIZE_Q
//         // (用于Q和计算)。 这里简化假设它们相等，或者 K Cache 已按 VEC_SIZE_Q
//         // 对齐。 如果不等，需要更复杂的加载和重排逻辑。
//         assert(VEC_SIZE_K_CACHE == VEC_SIZE_Q);  // 关键假设
// #pragma unroll
//         for (int elem_idx = 0; elem_idx < VEC_SIZE_Q; ++elem_idx) {
//           if constexpr (std::is_same<scalar_t, cache_t>::
//                             value) {  // 如果类型相同，直接赋值
//             k_vecs_reg[q_vec_idx].data[elem_idx] =
//                 k_cache_vec_loaded.data[elem_idx];
//           } else {  // 否则，进行类型转换 (例如 bfloat16 cache -> float
//           scalar
//                     // for computation)
//             from_float(k_vecs_reg[q_vec_idx].data[elem_idx],
//                        to_float(k_cache_vec_loaded.data[elem_idx]));
//           }
//         }
//       }  // 结束 K 子向量加载循环

//       // --- 计算 QK 点积 (已包含组内归约) ---
//       // q_vecs_shared[thread_in_computation_group_offset]
//       // 是当前线程在“计算小组”中负责的Q向量部分。 k_vecs_reg 是刚从 K Cache
//       // 加载并转换好的 K 向量部分。 Qk_dot::dot 返回的是已经过
//       // THREAD_GROUP_SIZE 个线程归约后的总和。
//       float qk_reduced_sum = Qk_dot<scalar_t, THREAD_GROUP_SIZE>::template
//       dot<
//           decltype(q_vecs_shared[0]),  // 类型是 Q_vec[NUM_Q_VECS_PER_THREAD]
//           decltype(k_vecs_reg),        // 类型是 K_vec[NUM_Q_VECS_PER_THREAD]
//           NUM_Q_VECS_PER_THREAD,       // 模板参数：每个线程的子向量数
//           VEC_SIZE_Q                   // 模板参数：每个子向量的元素数
//           >(q_vecs_shared[thread_in_computation_group_offset], k_vecs_reg);

//       // 应用缩放因子。
//       float qk_scaled = scale * qk_reduced_sum;

//       // 应用 ALiBi 位置偏置 (如果启用)。
//       if (alibi_slope != 0.f) {
//         qk_scaled += alibi_slope * (k_token_abs_logical_idx - seq_len + 1);
//       }

//       // --- 存储 Logits 到共享内存并更新局部最大值 ---
//       // 只有“计算小组”内的第一个线程 (thread_in_computation_group_offset ==
//       0)
//       // 负责写入共享内存中的 logits。 这是因为 QK 点积的结果 (一个标量
//       logit)
//       // 对应一个 (Q, K_token) 对。
//       if (thread_in_computation_group_offset == 0) {
//         // 计算当前 K Token 在当前分区内的逻辑索引。
//         const int k_token_idx_in_partition =
//             k_token_abs_logical_idx - start_token_idx;
//         // 检查掩码：如果当前 K Token 超出了序列的实际长度，则其 logit
//         // 应被屏蔽。
//         const bool is_masked_token = (k_token_abs_logical_idx >= seq_len);

//         if (k_token_idx_in_partition >= 0 &&
//             k_token_idx_in_partition < num_tokens_in_partition) {
//           logits_smem[k_token_idx_in_partition] =
//               is_masked_token ? -FLT_MAX : qk_scaled;
//           if (!is_masked_token) {  // 仅对未被屏蔽的 token 更新最大值
//             qk_max_local_thread = fmaxf(qk_max_local_thread, qk_scaled);
//           }
//         }
//       }
//     }  // 结束对 Block 内 K Tokens 的迭代 (NUM_K_TOKENS_PER_BLOCK_ITER)
//   }  // 结束对 Key Blocks 的迭代 (current_logical_block_idx)

//   // --- 6. 计算全局最大 Logit (qk_max_global) ---
//   // 此过程涉及两步归约：Warp 内归约 和 Warp 间归约。

//   // 步骤 6.1: Warp 内归约 qk_max_local_thread
//   // 每个线程 (主要是 thread_in_computation_group_offset == 0 的那些线程)
//   更新了
//   // qk_max_local_thread。 需要将这些值在 Warp 内归约。 为确保所有 lane
//   // 都有有效值参与，可以让非 leader 线程的 qk_max_local_thread 为 -FLT_MAX。
//   // 或者，更简单的是，所有线程都参与 shuffle，不影响结果因为 fmaxf 会忽略
//   // -FLT_MAX。
//   float qk_max_warp = qk_max_local_thread;  // 初始化为本线程计算的局部最大值
// #pragma unroll
//   for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
//     qk_max_warp = fmaxf(
//         qk_max_warp, __shfl_xor_sync(0xFFFFFFFF, qk_max_warp, mask,
//         WARP_SIZE));
//   }
//   // 此时，每个 Warp 的 Lane 0 持有该 Warp 内的 qk_max_local_thread
//   的最大值。

//   // 步骤 6.2: Warp 间归约，得到整个线程块的 qk_max_global
//   // 各个 Warp 的 Leader (Lane 0) 将其 qk_max_warp 写入共享内存
//   reduction_smem。 if (lane_idx == 0) {  // Warp Leaders
//     reduction_smem[warp_global_idx] = qk_max_warp;
//   }
//   __syncthreads();  // 确保所有 Warp Leaders 已写入共享内存。

//   float qk_max_global = -FLT_MAX;
//   // 由第一个 Warp (warp_global_idx == 0)
//   // 的线程负责从共享内存读取并完成最终归约。
//   if (warp_global_idx == 0) {
//     if (lane_idx < NUM_WARPS_IN_BLOCK) {  // 每个 Lane 读取一个 Warp 的最大值
//       qk_max_global = reduction_smem[lane_idx];
//     } else {
//       qk_max_global =
//           -FLT_MAX;  // 无效 Lane (如果 NUM_WARPS_IN_BLOCK < WARP_SIZE)
//     }
// #pragma unroll  // 在第一个 Warp 内归约从共享内存读到的各 Warp 最大值
//     for (int mask = NUM_WARPS_IN_BLOCK / 2; mask >= 1;
//          mask /= 2) {  // 注意归约宽度是 NUM_WARPS_IN_BLOCK
//       qk_max_global = fmaxf(
//           qk_max_global,
//           __shfl_xor_sync(0xFFFFFFFF, qk_max_global, mask,
//           NUM_WARPS_IN_BLOCK));
//     }
//     // 最终的 qk_max_global (整个线程块的最大 logit) 现在由 Warp 0, Lane 0
//     // 持有。
//   }
//   // 将 qk_max_global 广播给块内所有线程 (通过共享内存)。
//   if (warp_global_idx == 0 &&
//       lane_idx == 0) {  // 由线程块的第一个线程 (threadIdx.x == 0) 写入
//     reduction_smem[0] = qk_max_global;
//   }
//   __syncthreads();                    // 确保写入完成。
//   qk_max_global = reduction_smem[0];  // 所有线程从共享内存读取全局最大
//   logit。

//   // --- 7. 计算 Softmax Numerator (exp(logit - max_logit)) 和 Denominator
//   // (exp_sum) ---
//   float exp_sum_local_thread = 0.f;  // 每个线程计算的局部指数和
//   // 每个线程并行处理 num_tokens_in_partition / NUM_THREADS 个 logits。
//   for (int i = thread_global_idx; i < num_tokens_in_partition;
//        i += NUM_THREADS) {
//     // 从 logit 中减去 qk_max_global 可以提高数值稳定性，防止 exp() 上溢。
//     float val = __expf(logits_smem[i] - qk_max_global);
//     logits_smem[i] =
//         val;  // 更新共享内存中的 logits_smem 为 exp(logit - max_logit)
//         的值。
//     exp_sum_local_thread += val;  // 累加到当前线程的局部和。
//   }

//   // 使用 block_sum 工具函数归约所有线程的 exp_sum_local_thread，得到全局的
//   // exp_sum_global。 reduction_smem 的第二部分 (偏移 NUM_WARPS_IN_BLOCK)
//   用作
//   // block_sum 的工作区。
//   float exp_sum_global = block_sum<NUM_WARPS_IN_BLOCK>(
//       &reduction_smem[NUM_WARPS_IN_BLOCK], exp_sum_local_thread);

//   // --- 8. 计算最终 Softmax 概率 (P_ij = exp_val / exp_sum_global) ---
//   // 计算全局指数和的倒数，用于后续乘法代替除法，可能略微提高效率。
//   const float inv_exp_sum_global =
//       __fdividef(1.f, exp_sum_global + 1e-6f);  // 加 epsilon 防止除零。
//   // 每个线程并行更新其负责的 logits_smem 部分。
//   for (int i = thread_global_idx; i < num_tokens_in_partition;
//        i += NUM_THREADS) {
//     logits_smem[i] *=
//         inv_exp_sum_global;  // logits_smem 现在存储最终的 Softmax 概率
//         P_ij。
//   }
//   __syncthreads();  // 确保所有 Softmax 概率已计算并写回共享内存
//   logits_smem。

//   // --- (仅限 V2 版本) 存储中间结果: max_logits 和 exp_sums ---
//   // 如果是分区模式 (V2)，则由线程块的第一个线程 (threadIdx.x == 0)
//   将计算得到的
//   // qk_max_global 和 exp_sum_global 保存到全局内存的相应位置。
//   if (USE_PARTITIONING && (thread_global_idx == 0)) {
//     float* max_logits_out_ptr = max_logits +
//                                 seq_idx * num_q_heads * max_num_partitions +
//                                 head_idx * max_num_partitions +
//                                 partition_idx;
//     *max_logits_out_ptr = qk_max_global;

//     float* exp_sums_out_ptr = exp_sums +
//                               seq_idx * num_q_heads * max_num_partitions +
//                               head_idx * max_num_partitions + partition_idx;
//     *exp_sums_out_ptr = exp_sum_global;
//   }

//   // --- 9. 计算输出 O_i = sum_j (P_ij * V_j) ---
//   // V Cache 向量化加载配置。
//   // V_VEC_SIZE_CACHE: 一次从 V Cache 加载多少个 cache_t 元素。
//   constexpr int V_VEC_SIZE_CACHE = MIN(16 / sizeof(cache_t), BLOCK_SIZE);
//   using V_cache_vec =
//       VecType<cache_t, V_VEC_SIZE_CACHE>;  // V Cache 子向量类型。

//   // Softmax 概率 P_ij (存储在 logits_smem 中，为 float) 和输出 O (scalar_t)
//   // 的向量化配置。 V_VEC_SIZE_SCALAR: 处理 P_ij 和 V_j (转换为 scalar_t 后)
//   // 时，每个向量的元素数量。 通常希望它与 V_VEC_SIZE_CACHE
//   // 相匹配或兼容，以简化数据流。
//   constexpr int V_VEC_SIZE_SCALAR =
//       V_VEC_SIZE_CACHE;  // 简化假设：P和V的计算向量大小与V
//                          // Cache加载向量大小一致。
//   using P_vec =
//       VecType<scalar_t, V_VEC_SIZE_SCALAR>;  // P_ij (Softmax概率)
//       子向量类型。
//   using V_scalar_vec =
//       VecType<scalar_t, V_VEC_SIZE_SCALAR>;  // V_j (Value, 转换为
//       scalar_t后)
//                                              // 子向量类型。

//   // NUM_V_VECS_PER_BLOCK_ROW: 一个 BLOCK_SIZE 长度的 V 向量行 (对应一个 K/V
//   // Token)，
//   //                           需要多少个 V_VEC_SIZE_CACHE
//   大小的子向量来覆盖。 constexpr int NUM_V_VECS_PER_BLOCK_ROW = BLOCK_SIZE /
//   V_VEC_SIZE_CACHE; assert(BLOCK_SIZE % V_VEC_SIZE_CACHE == 0);  //
//   假设可以整除。

//   // NUM_HEAD_ROWS_PER_ITER_PER_WARP_LANE: 一个 Warp Lane
//   在一次内层迭代中，处理
//   // V 向量的多少“行” (对应 head_size 维度)。 一个 Warp (WARP_SIZE 个线程)
//   // 共同处理 V 的多行。 lane_idx % NUM_V_VECS_PER_BLOCK_ROW 决定了 Warp
//   // 内的线程处理 BLOCK_SIZE 中的哪一“列” (V_VEC_SIZE_CACHE 对齐)。 lane_idx
//   /
//   // NUM_V_VECS_PER_BLOCK_ROW 决定了 Warp 内的线程处理 HEAD_SIZE 中的哪一“行”
//   // (起始行)。
//   constexpr int NUM_HEAD_ROWS_PER_ITER_PER_WARP_LANE =
//       WARP_SIZE / NUM_V_VECS_PER_BLOCK_ROW;
//   assert(WARP_SIZE % NUM_V_VECS_PER_BLOCK_ROW == 0);  // 假设可以整除。

//   // NUM_HEAD_ROWS_PER_THREAD: 每个线程总共负责计算输出向量 O 的多少“行”
//   // (head_size 维度上的分片)。
//   constexpr int NUM_HEAD_ROWS_PER_THREAD =
//       DIVIDE_ROUND_UP(HEAD_SIZE, NUM_HEAD_ROWS_PER_ITER_PER_WARP_LANE);

//   // 累加器 (使用 float32 保证精度)，每个线程维护其负责的输出 O 的一部分。
//   float output_accumulators[NUM_HEAD_ROWS_PER_THREAD];
// #pragma unroll
//   for (int i = 0; i < NUM_HEAD_ROWS_PER_THREAD; i++) {
//     output_accumulators[i] = 0.f;
//   }

//   scalar_t zero_val_scalar;
//   zero(zero_val_scalar);  // 用于掩码操作的零值 (scalar_t 类型)。

//   // 迭代 Key/Value Blocks (与之前计算 QK 时相同的迭代逻辑)。
//   for (int current_logical_block_idx = start_block_idx + warp_global_idx;
//        current_logical_block_idx < end_block_idx;
//        current_logical_block_idx += NUM_WARPS_IN_BLOCK) {
//     // --- 块稀疏注意力: 跳过不相关的 Value Block (逻辑同 Key Block) ---
//     if constexpr (IS_BLOCK_SPARSE) {
//       int v_sparse_block_id =
//           current_logical_block_idx * BLOCK_SIZE / blocksparse_block_size;
//       if (!((v_sparse_block_id + bs_sliding_offset) % blocksparse_vert_stride
//       ==
//             0) &&
//           !((v_sparse_block_id >
//              q_sparse_block_id - blocksparse_local_blocks))) {
//         continue;  // 跳过此 Value Block。
//       }
//     }

//     const int64_t physical_block_number =
//         static_cast<int64_t>(block_table_for_seq[current_logical_block_idx]);

//     // 计算当前 Warp Lane 负责的 BLOCK_SIZE 内的“列”偏移 (V_VEC_SIZE_CACHE
//     // 对齐)。
//     const int col_offset_in_block_for_v =
//         (lane_idx % NUM_V_VECS_PER_BLOCK_ROW) * V_VEC_SIZE_CACHE;
//     // 当前“列”对应的 Token 在整个序列中的绝对逻辑索引。
//     const int v_token_abs_logical_idx =
//         current_logical_block_idx * BLOCK_SIZE + col_offset_in_block_for_v;
//     // 当前“列”对应的 Token 在当前分区内的逻辑索引 (相对于分区的
//     // start_token_idx)。
//     const int v_token_idx_in_partition =
//         v_token_abs_logical_idx - start_token_idx;

//     // 加载对应的 Softmax 概率 P_ij (存储在 logits_smem 中，类型为 float)。
//     // 需要加载 V_VEC_SIZE_SCALAR 个概率值，并转换为 scalar_t 类型存入
//     P_vec。 P_vec p_ij_vec;  // VecType<scalar_t, V_VEC_SIZE_SCALAR> if
//     (v_token_idx_in_partition >= 0 &&
//         v_token_idx_in_partition + V_VEC_SIZE_SCALAR <=
//             num_tokens_in_partition) {
//       // 从 float (logits_smem) 转换为 scalar_t 向量 (P_vec)。
//       // 临时 VecType<float, ...> 用于从 logits_smem 加载。
//       VecType<float, V_VEC_SIZE_SCALAR> temp_float_p_vec;
// #pragma unroll
//       for (int i = 0; i < V_VEC_SIZE_SCALAR; ++i) {
//         if (v_token_idx_in_partition + i < num_tokens_in_partition)  //
//         边界检查
//           temp_float_p_vec.data[i] = logits_smem[v_token_idx_in_partition +
//           i];
//         else
//           temp_float_p_vec.data[i] = 0.0f;  // 超出范围的概率视为0
//       }
//       convert_float_vec_to_scalar_vec<scalar_t, decltype(temp_float_p_vec),
//                                       V_VEC_SIZE_SCALAR>(p_ij_vec,
//                                                          temp_float_p_vec);
//     } else {  // 如果起始位置就超出范围，则整个 P_vec 为零。
// #pragma unroll
//       for (int i = 0; i < V_VEC_SIZE_SCALAR; ++i) zero(p_ij_vec.data[i]);
//     }

//     // V Cache 的指针基址，指向当前 (物理块, KV头)。
//     const cache_t* v_cache_base_ptr = v_cache +
//                                       physical_block_number * kv_block_stride
//                                       + kv_head_idx * kv_head_stride;

// // 迭代当前线程负责的 HEAD_SIZE 维度上的“行”。
// #pragma unroll
//     for (int head_row_iter = 0; head_row_iter < NUM_HEAD_ROWS_PER_THREAD;
//          ++head_row_iter) {
//       // 计算当前 Warp Lane 负责的 HEAD_SIZE 维度上的“行”索引。
//       const int head_row_idx =
//           (lane_idx / NUM_V_VECS_PER_BLOCK_ROW) +
//           head_row_iter * NUM_HEAD_ROWS_PER_ITER_PER_WARP_LANE;

//       if (head_row_idx < HEAD_SIZE) {  // 确保行索引在 head_size 范围内。
//         // 计算 V Cache 中数据的偏移量。
//         // V Cache 布局: [..., num_kv_heads, head_size, block_size]
//         // (按行主序存储 head_size 内的元素)
//         const int v_cache_offset =
//             head_row_idx * BLOCK_SIZE + col_offset_in_block_for_v;

//         // 从 V Cache 加载 cache_t 类型的向量。
//         V_cache_vec v_cache_vec_loaded = *reinterpret_cast<const
//         V_cache_vec*>(
//             v_cache_base_ptr + v_cache_offset);

//         // 将加载的 V_cache_vec (cache_t) 转换为 V_scalar_vec (scalar_t)。
//         V_scalar_vec v_j_vec;  // VecType<scalar_t, V_VEC_SIZE_SCALAR>
// #pragma unroll
//         for (int elem_idx = 0; elem_idx < V_VEC_SIZE_SCALAR; ++elem_idx) {
//           if constexpr (std::is_same<scalar_t, cache_t>::value) {
//             v_j_vec.data[elem_idx] = v_cache_vec_loaded.data[elem_idx];
//           } else {
//             from_float(v_j_vec.data[elem_idx],
//                        to_float(v_cache_vec_loaded.data[elem_idx]));
//           }
//         }

//         // 处理最后一个 Block 的边界情况 (对 Value进行掩码)。
//         // 如果是序列的最后一个逻辑 Block，并且加载的 V Token
//         // 超出了实际序列长度，
//         则应将其值置为零，以避免无效数据影响计算结果。 if
//         (current_logical_block_idx ==
//             num_seq_blocks - 1) {  // 是否为最后一个逻辑块
// #pragma unroll
//           for (int j = 0; j < V_VEC_SIZE_SCALAR; ++j) {
//             if (v_token_abs_logical_idx + j >=
//                 seq_len) {  // 检查每个元素是否超出序列长度
//               v_j_vec.data[j] = zero_val_scalar;
//             }
//           }
//         }
//         // 累加 P_ij * V_j 到当前线程的对应累加器。
//         output_accumulators[head_row_iter] +=
//             dot_vectors<scalar_t, scalar_t, V_VEC_SIZE_SCALAR>(p_ij_vec,
//                                                                v_j_vec);
//       }
//     }  // 结束对 head_size “行”的迭代
//   }  // 结束对 Key/Value Blocks 的迭代

// // --- 10. Warp 内归约输出累加器 output_accumulators ---
// // 每个 output_accumulators[i] 累加了来自 BLOCK_SIZE 内不同“列” (由 lane_idx
// %
// // NUM_V_VECS_PER_BLOCK_ROW 区分) 的贡献。 需要将这些贡献在 Warp 内的
// // NUM_V_VECS_PER_BLOCK_ROW 个 Lane 之间进行归约。
// #pragma unroll
//   for (int i = 0; i < NUM_HEAD_ROWS_PER_THREAD; i++) {
//     float acc_val = output_accumulators[i];
// #pragma unroll
//     for (int mask = NUM_V_VECS_PER_BLOCK_ROW / 2; mask >= 1; mask /= 2) {
//       // __shfl_xor_sync 的 width 参数应为归约组的大小，这里是
//       // NUM_V_VECS_PER_BLOCK_ROW。
//       acc_val +=
//           __shfl_xor_sync(0xFFFFFFFF, acc_val, mask,
//           NUM_V_VECS_PER_BLOCK_ROW);
//     }
//     output_accumulators[i] =
//         acc_val;  // 归约后的结果存储在每个 NUM_V_VECS_PER_BLOCK_ROW 组的
//         Lane 0
//                   // 中。
//   }

//   // 同步，因为 logits_smem (即 shared_mem_softmax_area 的一部分)
//   // 的共享内存空间将被复用为输出的 Warp 间归约临时存储。
//   __syncthreads();

//   // --- 11. Warp 间归约输出累加器 output_accumulators ---
//   // 使用 logits_smem (现在可以安全复用) 作为 Warp 间归约的临时存储区。
//   // 大小需要: NUM_WARPS_IN_BLOCK * HEAD_SIZE 个 float。确保共享内存足够。
//   float* output_reduction_smem =
//       reinterpret_cast<float*>(shared_mem_softmax_area);

// // 这是一个树形归约 (tree-based reduction) 过程。
// // 每次迭代，一半的 Warp (高地址的) 将其数据写入共享内存，
// // 另一半 Warp (低地址的) 从共享内存读取并累加到自己的累加器中。
// #pragma unroll
//   for (int num_active_warps_for_reduce = NUM_WARPS_IN_BLOCK;
//        num_active_warps_for_reduce > 1; num_active_warps_for_reduce /= 2) {
//     int mid_point_warps =
//         (num_active_warps_for_reduce + 1) / 2;  // 处理奇数 Warp 数的情况

//     // 位于 [mid_point_warps, num_active_warps_for_reduce) 区间的 "源" Warp
//     将其
//     // output_accumulators 写入共享内存。
//     if (warp_global_idx >= mid_point_warps &&
//         warp_global_idx < num_active_warps_for_reduce) {
//       // 计算在共享内存中的目标地址。 (warp_global_idx - mid_point_warps)
//       // 得到的是相对于目标区域的 Warp 索引。
//       float* dst_smem_ptr =
//           &output_reduction_smem[(warp_global_idx - mid_point_warps) *
//                                  HEAD_SIZE];
// #pragma unroll
//       for (int i = 0; i < NUM_HEAD_ROWS_PER_THREAD; i++) {
//         const int head_row_idx = (lane_idx / NUM_V_VECS_PER_BLOCK_ROW) +
//                                  i * NUM_HEAD_ROWS_PER_ITER_PER_WARP_LANE;
//         // 只有每个 NUM_V_VECS_PER_BLOCK_ROW 组的 Lane 0 (持有 Warp 内归约后
//         // output_accumulators[i] 的线程) 负责写入。
//         if (head_row_idx < HEAD_SIZE &&
//             (lane_idx % NUM_V_VECS_PER_BLOCK_ROW == 0)) {
//           dst_smem_ptr[head_row_idx] = output_accumulators[i];
//         }
//       }
//     }
//     __syncthreads();  // 确保所有源 Warp 已写入共享内存。

//     // 位于 [0, mid_point_warps) 区间的 "目标" Warp 从共享内存读取并累加。
//     if (warp_global_idx < mid_point_warps) {
//       // 确保有对应的源数据可供读取。
//       // (num_active_warps_for_reduce - mid_point_warps) 是实际源 Warp
//       的数量。 if (warp_global_idx < (num_active_warps_for_reduce -
//       mid_point_warps)) {
//         const float* src_smem_ptr =
//             &output_reduction_smem[warp_global_idx * HEAD_SIZE];
// #pragma unroll
//         for (int i = 0; i < NUM_HEAD_ROWS_PER_THREAD; i++) {
//           const int head_row_idx = (lane_idx / NUM_V_VECS_PER_BLOCK_ROW) +
//                                    i * NUM_HEAD_ROWS_PER_ITER_PER_WARP_LANE;
//           if (head_row_idx < HEAD_SIZE &&
//               (lane_idx % NUM_V_VECS_PER_BLOCK_ROW == 0)) {
//             output_accumulators[i] += src_smem_ptr[head_row_idx];
//           }
//         }
//       }
//     }
//     __syncthreads();  // 确保所有目标 Warp 已完成累加。
//   }  // 结束 Warp 间归约循环。

//   // --- 12. 最终结果写入全局内存 ---
//   // 此时，Warp 0 (warp_global_idx == 0) 的线程持有最终的累加结果
//   // output_accumulators。 具体来说，是 Warp 0 内，每个
//   NUM_V_VECS_PER_BLOCK_ROW
//   // 组的 Lane 0 线程， 持有其对应 head_row_idx 的最终结果。
//   if (warp_global_idx == 0) {
//     scalar_t* out_global_ptr;
//     if (USE_PARTITIONING) {   // V2 版本: 输出到临时的 tmp_out 张量
//       out_global_ptr = out +  // `out` 此时指向 tmp_out
//                        seq_idx * num_q_heads * max_num_partitions * HEAD_SIZE
//                        + head_idx * max_num_partitions * HEAD_SIZE +
//                        partition_idx * HEAD_SIZE;
//     } else {  // V1 版本: 直接输出到最终的 out 张量
//       out_global_ptr =
//           out + seq_idx * num_q_heads * HEAD_SIZE + head_idx * HEAD_SIZE;
//     }

// // Warp 0 内的线程并行写入其负责的 head_size 部分。
// #pragma unroll
//     for (int i = 0; i < NUM_HEAD_ROWS_PER_THREAD; i++) {
//       const int head_row_idx = (lane_idx / NUM_V_VECS_PER_BLOCK_ROW) +
//                                i * NUM_HEAD_ROWS_PER_ITER_PER_WARP_LANE;
//       if (head_row_idx < HEAD_SIZE &&
//           (lane_idx % NUM_V_VECS_PER_BLOCK_ROW == 0)) {
//         from_float(*(out_global_ptr + head_row_idx), output_accumulators[i]);
//       }
//     }
//   }
// }  // 结束 paged_attention_kernel

// // --- Global Kernel Wrappers ---
// // 这些是 __global__ 函数，可以从主机端代码启动。
// // 它们内部调用核心的 __device__ 函数 paged_attention_kernel。

// /**
//  * @brief Paged Attention V1 (无分区) 的全局 CUDA 核函数。
//  * CUDA Kernel 启动配置:
//  * - Grid: (num_attention_heads, num_sequences, 1)
//  * - Block: (NUM_THREADS)
//  * - Dynamic Shared Memory: 计算 paged_attention_kernel 所需的共享内存大小。
//  */
// template <typename scalar_t, typename cache_t, int HEAD_SIZE, int BLOCK_SIZE,
//           int NUM_THREADS, bool IS_BLOCK_SPARSE>
// __global__ void paged_attention_v1_kernel(
//     scalar_t* __restrict__ out,  // 输出: [num_seqs, num_heads, head_size]
//     const scalar_t* __restrict__ q, const cache_t* __restrict__ k_cache,
//     const cache_t* __restrict__ v_cache, const int num_kv_heads,
//     const float scale, const int* __restrict__ block_tables,
//     const int* __restrict__ seq_lens, const int max_num_blocks_per_seq,
//     const float* __restrict__ alibi_slopes, const int q_stride,
//     const int kv_block_stride, const int kv_head_stride, const int tp_rank,
//     const int blocksparse_local_blocks, const int blocksparse_vert_stride,
//     const int blocksparse_block_size, const int
//     blocksparse_head_sliding_step) {
//   // 调用核心设备函数，PARTITION_SIZE 设置为 0 表示 V1 (无分区) 模式。
//   paged_attention_kernel<scalar_t, cache_t, HEAD_SIZE, BLOCK_SIZE,
//   NUM_THREADS,
//                          IS_BLOCK_SPARSE, 0 /* PARTITION_SIZE = 0 for v1 */>(
//       /*exp_sums=*/nullptr,    // V1 无此输出
//       /*max_logits=*/nullptr,  // V1 无此输出
//       out,                     // 直接写入最终输出
//       q, k_cache, v_cache, num_kv_heads, scale, block_tables, seq_lens,
//       max_num_blocks_per_seq, alibi_slopes, q_stride, kv_block_stride,
//       kv_head_stride, tp_rank, blocksparse_local_blocks,
//       blocksparse_vert_stride, blocksparse_block_size,
//       blocksparse_head_sliding_step);
// }

// /**
//  * @brief Paged Attention V2 (带分区，计算部分结果) 的全局 CUDA 核函数。
//  * 此核函数计算每个分区的中间结果，并将其存储到 tmp_out, exp_sums,
//  max_logits。
//  * 后续需要 paged_attention_v2_reduce_kernel 来合并这些分区结果。
//  * CUDA Kernel 启动配置:
//  * - Grid: (num_attention_heads, num_sequences, max_num_partitions)
//  * - Block: (NUM_THREADS)
//  * - Dynamic Shared Memory: 计算 paged_attention_kernel 所需的共享内存大小。
//  */
// template <typename scalar_t, typename cache_t, int HEAD_SIZE, int BLOCK_SIZE,
//           int NUM_THREADS, bool IS_BLOCK_SPARSE,
//           int PARTITION_SIZE>  // PARTITION_SIZE > 0 for v2
// __global__ void paged_attention_v2_kernel(
//     float* __restrict__ exp_sums,    // 输出: [num_seqs, num_heads,
//                                      // max_num_partitions]
//     float* __restrict__ max_logits,  // 输出: [num_seqs, num_heads,
//                                      // max_num_partitions]
//     scalar_t* __restrict__ tmp_out,  // 输出 (临时): [num_seqs, num_heads,
//                                      // max_num_partitions, head_size]
//     const scalar_t* __restrict__ q, const cache_t* __restrict__ k_cache,
//     const cache_t* __restrict__ v_cache, const int num_kv_heads,
//     const float scale, const int* __restrict__ block_tables,
//     const int* __restrict__ seq_lens, const int max_num_blocks_per_seq,
//     const float* __restrict__ alibi_slopes, const int q_stride,
//     const int kv_block_stride, const int kv_head_stride, const int tp_rank,
//     const int blocksparse_local_blocks, const int blocksparse_vert_stride,
//     const int blocksparse_block_size, const int
//     blocksparse_head_sliding_step) {
//   // 调用核心设备函数，传入 PARTITION_SIZE。
//   paged_attention_kernel<scalar_t, cache_t, HEAD_SIZE, BLOCK_SIZE,
//   NUM_THREADS,
//                          IS_BLOCK_SPARSE, PARTITION_SIZE>(
//       exp_sums, max_logits, tmp_out, q, k_cache, v_cache, num_kv_heads,
//       scale, block_tables, seq_lens, max_num_blocks_per_seq, alibi_slopes,
//       q_stride, kv_block_stride, kv_head_stride, tp_rank,
//       blocksparse_local_blocks, blocksparse_vert_stride,
//       blocksparse_block_size, blocksparse_head_sliding_step);
// }

// /**
//  * @brief Paged Attention V2 Reduce Kernel (合并 V2 的分区结果)。
//  * 此核函数读取 paged_attention_v2_kernel 计算的各分区中间结果
//  * (tmp_out, exp_sums, max_logits)，并将其合并得到最终的注意力输出。
//  * CUDA Kernel 启动配置:
//  * - Grid: (num_attention_heads, num_sequences)
//  * - Block: (NUM_THREADS)
//  * - Dynamic Shared Memory: 计算此 reduce kernel 所需的共享内存大小。
//  */
// template <typename scalar_t, int HEAD_SIZE, int NUM_THREADS,
//           int PARTITION_SIZE>  // PARTITION_SIZE > 0 for v2
// __global__ void paged_attention_v2_reduce_kernel(
//     scalar_t* __restrict__ out,  // 最终输出: [num_seqs, num_heads,
//     head_size] const float* __restrict__ exp_sums,    // 输入: [num_seqs,
//     num_heads,
//                                            // max_num_partitions]
//     const float* __restrict__ max_logits,  // 输入: [num_seqs, num_heads,
//                                            // max_num_partitions]
//     const scalar_t* __restrict__ tmp_out,  // 输入 (临时): [num_seqs,
//     num_heads,
//                                            // max_num_partitions, head_size]
//     const int* __restrict__ seq_lens,      // 输入: [num_seqs]
//     const int max_num_partitions  // 最大分区数 (gridDim.z of v2_kernel)
// ) {
//   // --- 1. 计算线程和块索引，确定处理的序列和头 ---
//   const int num_q_heads = gridDim.x;      // 总的查询头数量
//   const int head_idx = blockIdx.x;        // 当前线程块处理的注意力头索引
//   const int seq_idx = blockIdx.y;         // 当前线程块处理的序列索引
//   const int seq_len = seq_lens[seq_idx];  // 当前序列的实际长度

//   // 计算当前序列实际有多少个活动的分区。
//   const int num_active_partitions = DIVIDE_ROUND_UP(seq_len, PARTITION_SIZE);

//   // 如果只有一个活动分区，则无需进行复杂的归约操作，直接从 tmp_out
//   拷贝数据到
//   // out。
//   if (num_active_partitions == 1) {
//     scalar_t* out_ptr_global =
//         out + seq_idx * num_q_heads * HEAD_SIZE + head_idx * HEAD_SIZE;
//     const scalar_t* tmp_out_ptr_partition0 =
//         tmp_out + seq_idx * num_q_heads * max_num_partitions * HEAD_SIZE +
//         head_idx * max_num_partitions * HEAD_SIZE +
//         0 * HEAD_SIZE;  // partition_idx = 0
//     // 线程块内的线程并行拷贝 HEAD_SIZE 个元素。
//     for (int i = threadIdx.x; i < HEAD_SIZE; i += blockDim.x /*NUM_THREADS*/)
//     {
//       out_ptr_global[i] = tmp_out_ptr_partition0[i];
//     }
//     return;  // 提前结束该线程块的执行。
//   }

//   // --- 2. 准备共享内存和线程参数 ---
//   constexpr int NUM_WARPS_IN_REDUCE_BLOCK = NUM_THREADS / WARP_SIZE;
//   const int warp_idx_reduce =
//       threadIdx.x / WARP_SIZE;  // 当前线程所属 Warp 在块内的索引
//   const int lane_idx_reduce =
//       threadIdx.x % WARP_SIZE;  // 当前线程在其 Warp 内的 Lane 索引

//   // 动态共享内存，用于存储各分区的 max_logits、重缩放后的
//   // exp_sums，以及归约操作的工作区。 所需空间:
//   // - shared_max_logits: num_active_partitions * sizeof(float)
//   // - shared_rescaled_exp_sums: num_active_partitions * sizeof(float)
//   // - reduction_workspace: (至少) 2 * NUM_WARPS_IN_REDUCE_BLOCK *
//   sizeof(float) extern __shared__ char shared_mem_reduce_area[]; float*
//   shared_max_logits = reinterpret_cast<float*>(shared_mem_reduce_area);
//   float* shared_rescaled_exp_sums = reinterpret_cast<float*>(
//       shared_mem_reduce_area + num_active_partitions * sizeof(float));
//   float* reduction_workspace = reinterpret_cast<float*>(
//       shared_mem_reduce_area + 2 * num_active_partitions * sizeof(float));

//   // --- 3. 加载各分区的 max_logits 到共享内存，并计算全局 max_logit ---
//   // 指向当前 (seq_idx, head_idx) 的 max_logits 数据的起始位置。
//   const float* max_logits_base_ptr =
//       max_logits + seq_idx * num_q_heads * max_num_partitions +
//       head_idx * max_num_partitions;
//   float current_max_logit_global_thread = -FLT_MAX;  // 当前线程的局部最大
//   logit
//   // 线程块内的线程并行加载各分区的 max_logit 到共享内存 shared_max_logits。
//   // 同时，每个线程在其处理的分区中更新 current_max_logit_global_thread。
//   for (int p_iter_idx = threadIdx.x; p_iter_idx < num_active_partitions;
//        p_iter_idx += NUM_THREADS) {
//     float l = max_logits_base_ptr[p_iter_idx];
//     shared_max_logits[p_iter_idx] = l;
//     current_max_logit_global_thread = fmaxf(current_max_logit_global_thread,
//     l);
//   }
//   __syncthreads();  // 确保所有分区的 max_logit 已加载到共享内存。

// // 归约得到全局 max_logit (在所有活动分区中)。
// // (与 paged_attention_kernel 中的 max logit 归约逻辑类似)
// // Warp 内归约
// #pragma unroll
//   for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
//     current_max_logit_global_thread =
//         fmaxf(current_max_logit_global_thread,
//               __shfl_xor_sync(0xFFFFFFFF, current_max_logit_global_thread,
//               mask,
//                               WARP_SIZE));
//   }
//   // Warp 间归约
//   if (lane_idx_reduce == 0)
//     reduction_workspace[warp_idx_reduce] = current_max_logit_global_thread;
//   __syncthreads();

//   float final_max_logit_global = -FLT_MAX;
//   if (warp_idx_reduce == 0) {  // 第一个 Warp 执行最终的 Warp 间归约
//     if (lane_idx_reduce < NUM_WARPS_IN_REDUCE_BLOCK)
//       final_max_logit_global = reduction_workspace[lane_idx_reduce];
// #pragma unroll
//     for (int mask = NUM_WARPS_IN_REDUCE_BLOCK / 2; mask >= 1; mask /= 2) {
//       final_max_logit_global =
//           fmaxf(final_max_logit_global,
//                 __shfl_xor_sync(0xFFFFFFFF, final_max_logit_global, mask,
//                                 NUM_WARPS_IN_REDUCE_BLOCK));
//     }
//   }
//   // 广播全局 max_logit 给块内所有线程
//   if (warp_idx_reduce == 0 && lane_idx_reduce == 0)
//     reduction_workspace[0] = final_max_logit_global;
//   __syncthreads();
//   final_max_logit_global = reduction_workspace[0];

//   // --- 4. 加载各分区的 exp_sums，根据全局 max_logit
//   进行重缩放，并计算全局总的
//   // exp_sum --- 指向当前 (seq_idx, head_idx) 的 exp_sums 数据的起始位置。
//   const float* exp_sums_base_ptr = exp_sums +
//                                    seq_idx * num_q_heads * max_num_partitions
//                                    + head_idx * max_num_partitions;
//   float current_total_exp_sum_global_thread =
//       0.0f;  // 当前线程的局部重缩放后指数和
//   // 线程块内的线程并行处理各分区的 exp_sum。
//   for (int p_iter_idx = threadIdx.x; p_iter_idx < num_active_partitions;
//        p_iter_idx += NUM_THREADS) {
//     float partition_max_logit =
//         shared_max_logits[p_iter_idx];  // 从共享内存读取该分区的 max_logit
//     // 重缩放公式: new_exp_sum[p] = old_exp_sum[p] * exp(max_logit[p] -
//     // final_max_logit_global)
//     float rescaled_exp_sum =
//         exp_sums_base_ptr[p_iter_idx] *
//         __expf(partition_max_logit - final_max_logit_global);
//     shared_rescaled_exp_sums[p_iter_idx] = rescaled_exp_sum;  // 存入共享内存
//     current_total_exp_sum_global_thread +=
//         rescaled_exp_sum;  // 累加到线程局部和
//   }
//   __syncthreads();  // 确保所有重缩放后的 exp_sum 已计算并写入共享内存。

//   // 使用 block_sum 工具函数归约得到全局总的重缩放后 exp_sum。
//   // reduction_workspace 的第二部分用作 block_sum 的工作区。
//   float final_total_exp_sum_global = block_sum<NUM_WARPS_IN_REDUCE_BLOCK>(
//       &reduction_workspace[NUM_WARPS_IN_REDUCE_BLOCK],
//       current_total_exp_sum_global_thread);
//   const float inv_final_total_exp_sum_global = __fdividef(
//       1.0f, final_total_exp_sum_global + 1e-6f);  // 加 epsilon 防除零。

//   // --- 5. 聚合各分区的临时输出 tmp_out 到最终的输出 out ---
//   // 最终输出公式: out_final[h] = sum_over_partitions ( tmp_out[p][h] *
//   // shared_rescaled_exp_sums[p] ) * inv_final_total_exp_sum_global 指向当前
//   // (seq_idx, head_idx) 的 tmp_out 数据的起始位置。
//   const scalar_t* tmp_out_base_ptr =
//       tmp_out + seq_idx * num_q_heads * max_num_partitions * HEAD_SIZE +
//       head_idx * max_num_partitions * HEAD_SIZE;
//   // 指向最终输出 out 的对应位置。
//   scalar_t* out_final_ptr =
//       out + seq_idx * num_q_heads * HEAD_SIZE + head_idx * HEAD_SIZE;

// // 线程块内的线程并行计算最终输出 out 的每个元素 (在 HEAD_SIZE 维度上)。
// #pragma unroll
//   for (int h_dim_idx = threadIdx.x; h_dim_idx < HEAD_SIZE;
//        h_dim_idx += NUM_THREADS) {
//     float final_output_accumulator = 0.0f;  // 用于累加的浮点累加器
//     // 遍历所有活动分区
//     for (int p_idx = 0; p_idx < num_active_partitions; ++p_idx) {
//       // 从 tmp_out 中获取分区 p 的第 h_dim_idx 个元素值。
//       scalar_t tmp_out_val_for_partition =
//           tmp_out_base_ptr[p_idx * HEAD_SIZE + h_dim_idx];
//       // 累加: (分区输出值 * 分区重缩放后exp_sum)
//       final_output_accumulator +=
//           to_float(tmp_out_val_for_partition) *
//           shared_rescaled_exp_sums[p_idx];
//     }
//     // 乘以全局总exp_sum的倒数，完成加权平均。
//     final_output_accumulator *= inv_final_total_exp_sum_global;
//     // 将计算得到的浮点结果转换回 scalar_t 类型并存入最终输出。
//     from_float(out_final_ptr[h_dim_idx], final_output_accumulator);
//   }
// }  // 结束 paged_attention_v2_reduce_kernel

// }  // namespace vllm

// // 清理之前定义的宏 (如果后续代码不需要这些宏，这是一种良好的编程习惯)
// #undef WARP_SIZE
// #undef MAX
// #undef MIN
// #undef DIVIDE_ROUND_UP

// #endif  // VLLM_PAGED_ATTENTION_KERNELS_HPP
