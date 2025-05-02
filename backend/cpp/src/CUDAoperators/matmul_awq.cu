// matmul_awq.cu
// AWQ量化矩阵乘法CUDA实现 - v14 (Optimized Kernel based on v12 structure)
// 目的: 实现AWQ 4bit量化权重的矩阵乘法, 处理AWQ特定的内部顺序
// 处理: Input[M, K], QWeight[K, N/8], Scales[NumGroups, N], Zeros[NumGroups,
// N/8] 输出 = Input * Dequant(QWeight, Scales, Zeros) #define DEBUG_AWQ
#include <cuda_bf16.h>  // for __nv_bfloat16
#include <cuda_fp16.h>  // for __half
#include <cuda_runtime.h>
#include <stdint.h>  // for int32_t, uint32_t

#include <iomanip>  // for std::setw, std::setprecision
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cudaOP.cuh"  // Assuming this contains your Tensor class definition

namespace cuda_OP {
// Helper function to print tensor sizes (reuse from previous version)
inline std::string format_sizes(const std::vector<size_t> &sizes) {
  std::stringstream ss;
  ss << "(";
  for (size_t i = 0; i < sizes.size(); ++i) {
    ss << sizes[i] << (i == sizes.size() - 1 ? "" : ", ");
  }
  ss << ")";
  return ss.str();
}

// Helper function to print tensor values for debugging (reuse)
template <typename T>
void debug_print_tensor(const Tensor<T> &tensor, const std::string &name,
                        int max_elements = 10) {
#ifdef DEBUG_AWQ
  std::cout << "Tensor " << name << " " << format_sizes(tensor.sizes()) << ":"
            << std::endl;
  std::vector<T> host_data(tensor.numel());
  cudaMemcpy(host_data.data(), tensor.data_ptr(), tensor.numel() * sizeof(T),
             cudaMemcpyDeviceToHost);
  int num_to_print = std::min(static_cast<int>(tensor.numel()), max_elements);
  for (int i = 0; i < num_to_print; ++i) {
    // Handle different types for printing
    float val;
    if constexpr (std::is_same_v<T, float>) {
      val = host_data[i];
    } else if constexpr (std::is_same_v<T, __half>) {
      val = __half2float(host_data[i]);
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      val = __bfloat162float(host_data[i]);
    } else {
      val = static_cast<float>(host_data[i]);  // Fallback for int etc.
    }
    std::cout << "  [" << i << "] = " << std::fixed << std::setprecision(4)
              << val << std::endl;
  }
  if (tensor.numel() > max_elements) {
    std::cout << "  ... (and " << tensor.numel() - max_elements
              << " more elements)" << std::endl;
  }
#endif
}
//----------------------------------------------------------------------------
// 常量和映射定义
//----------------------------------------------------------------------------
constexpr int BITS = 4;
constexpr int PACK_FACTOR = 32 / BITS;  // = 8

// AWQ 内部物理顺序映射: logical_inner_idx -> physical_inner_idx used for bit
// shifts
__constant__ const int LOGICAL_TO_PHYSICAL_INNER_IDX[PACK_FACTOR] = {
    0, 4, 1, 5, 2, 6, 3, 7};

//----------------------------------------------------------------------------
// Optimized CUDA Kernel (AWQ Dequant + Matmul Core with Shared Memory Tiling)
//----------------------------------------------------------------------------
template <typename T, typename ScaleType, int TILE_K_ = 32, int BLOCK_N_ = 256,
          int BLOCK_M_ = 1>
__global__ void matmul_awq_kernel_optimized(
    const T *__restrict__ inp,  // 输入激活: [M, K]
    const int32_t
        *__restrict__ qwt,  // 量化权重: [K, N / PACK_FACTOR] (AWQ Order)
    const ScaleType
        *__restrict__ scl,  // 缩放因子: [NumGroups, N] (Logical Order)
    const int32_t *__restrict__ zos,  // 量化零点: [NumGroups, N / PACK_FACTOR]
                                      // (AWQ Order)
    T *__restrict__ out,  // 输出结果: [M, N]
    const int M, const int K, const int N, const int group_size,
    const T *__restrict__ bias)  // 可选的偏置: [N]
{
  // --- Compile-time constants ---
  constexpr int TILE_K = TILE_K_;
  constexpr int BLOCK_N = BLOCK_N_;  // Should match blockDim.x
  constexpr int BLOCK_M = BLOCK_M_;  // Assumes 1 row per block

  // --- Shared Memory Declaration ---
  // Cache for input activations (adjust size if BLOCK_M > 1)
  __shared__ T sh_inp[TILE_K];  // Size TILE_K for BLOCK_M = 1
  // Cache for packed weights [TILE_K, BLOCK_N / PACK_FACTOR]
  __shared__ int32_t sh_qwt[TILE_K][BLOCK_N / PACK_FACTOR];

  // --- Thread Indexing ---
  const int tidx =
      threadIdx.x;  // Thread index within the block (0 to BLOCK_N-1)

  // Logical output row index (one row per block)
  const int m = blockIdx.y;
  // Logical output column index this thread is responsible for
  const int n = blockIdx.x * BLOCK_N + tidx;

  // Starting column index for the packed data handled by this block
  const int block_packed_col_start = blockIdx.x * (BLOCK_N / PACK_FACTOR);
  // Column index within the block's packed data tile (identifies which packed
  // int32 column in shared memory the thread reads)
  const int block_packed_col_offset = tidx / PACK_FACTOR;
  // Packed column index this thread needs to load *from global memory* during
  // the load phase (used when tidx < BLOCK_N/PACK_FACTOR for loading weights)
  // const int packed_col_to_load = block_packed_col_start + tidx; // potential
  // load index if 1 thread loads 1 packed int

  // --- Pre-calculate AWQ Reordering Info (Constant per thread for the logical
  // output column n) --- Logical index within the 8-element pack
  const int logical_inner_idx = n % PACK_FACTOR;
  // Physical index within the 8-element pack using AWQ map
  const int physical_inner_col_idx =
      LOGICAL_TO_PHYSICAL_INNER_IDX[logical_inner_idx];
  // Bit shift amount for unpacking based on the physical location
  const int bit_shift = physical_inner_col_idx * BITS;

  // --- Accumulator Initialization ---
  float acc = 0.0f;

  // --- Check Grid Boundaries (Early Exit for out-of-bounds threads) ---
  // Check 'm' boundary first, as it determines the row validity
  if (m >= M) return;
  // Check 'n' boundary for threads targeting columns beyond N
  // Note: Threads with n >= N still participate in loading if their tidx
  // contributes But they won't compute or write output. The check is done
  // before accumulation/write.

  // --- Main Loop over K dimension in Tiles ---
  for (int k_base = 0; k_base < K; k_base += TILE_K) {
    // --- Load Data into Shared Memory ---
    __syncthreads();  // Sync before loading the next tile

    // 1. Load Input Activation Tile (sh_inp) cooperatively
    // Each thread loads one element if TILE_K <= BLOCK_N.
    // This simple version assumes TILE_K <= BLOCK_N and BLOCK_M = 1.
    if (tidx < TILE_K) {
      const int k_load = k_base + tidx;
      // Check boundary for K dimension
      if (k_load < K) {
        sh_inp[tidx] = inp[m * K + k_load];
      } else {
        // Padding with zero if k_load exceeds K
        sh_inp[tidx] = static_cast<T>(0.0f);
      }
    }

    // 2. Load Packed Quantized Weights Tile (sh_qwt) cooperatively
    // Each thread loads values into the sh_qwt tile.
    // Stride through the tile elements using thread index.
    // Ensure loads are coalesced as much as possible.
    for (int load_idx = tidx; load_idx < TILE_K * (BLOCK_N / PACK_FACTOR);
         load_idx += BLOCK_N) {
      const int tk_load =
          load_idx /
          (BLOCK_N / PACK_FACTOR);  // Row index in shared tile (0..TILE_K-1)
      const int packed_col_in_tile =
          load_idx % (BLOCK_N / PACK_FACTOR);  // Column index in shared tile

      const int k_global = k_base + tk_load;  // Global K index for this element
      // Global packed column index corresponding to the column in the shared
      // tile
      const int packed_col_global_for_load =
          block_packed_col_start + packed_col_in_tile;

      // Check boundaries before loading from global memory
      if (k_global < K && packed_col_global_for_load < (N / PACK_FACTOR)) {
        sh_qwt[tk_load][packed_col_in_tile] =
            qwt[k_global * (N / PACK_FACTOR) + packed_col_global_for_load];
      } else {
        // Padding with zero if outside bounds
        sh_qwt[tk_load][packed_col_in_tile] = 0;
      }
    }

    // Wait for all threads in the block to finish loading into shared memory
    __syncthreads();

    // --- Computation using Shared Memory ---
    // Early exit for threads that are completely outside the N boundary
    if (n >= N) continue;  // Skip computation and writing for these threads

    int current_group_idx = -1;
    int32_t current_packed_zos =
        0;  // Cache the packed zero for the current group
    ScaleType current_scale_val = static_cast<ScaleType>(
        0.0f);  // Cache the scale for the current group and column n

    // Loop over K within the loaded tile
    // #pragma unroll // Optional: Consider unrolling for small TILE_K, profile
    // needed
    for (int tk = 0; tk < TILE_K; ++tk) {
      const int k =
          k_base +
          tk;  // Current global K index
               // Check if k is out of bounds (due to padding in the last tile)
      if (k >= K) break;

      // 1. Get Input Activation from Shared Memory
      const float inp_val = static_cast<float>(sh_inp[tk]);

      // 2. Get Quantization Parameters (Scale & Zero) - Load only when group
      // changes
      const int group_idx = k / group_size;
      if (group_idx != current_group_idx) {
        current_group_idx = group_idx;
        const int num_groups =
            (K + group_size - 1) /
            group_size;  // Calculate total number of groups safely

        // Calculate global index for packed zeros (shared by threads handling
        // the same packed col)
        const int zos_packed_col_global =
            block_packed_col_start + block_packed_col_offset;
        const int zos_idx =
            group_idx * (N / PACK_FACTOR) + zos_packed_col_global;

        // Boundary check for zeros access
        if (group_idx < num_groups &&
            zos_packed_col_global < (N / PACK_FACTOR)) {
          current_packed_zos = zos[zos_idx];
        } else {
          current_packed_zos = 0;  // Use zero padding if out of bounds
        }

        // Calculate global index for scales (specific to logical column n)
        const int scl_idx = group_idx * N + n;  // Use logical 'n'

        // Boundary check for scales access (n already checked, check group_idx)
        if (group_idx < num_groups) {  // n < N check is implicitly done by
                                       // thread boundary check
          current_scale_val = scl[scl_idx];
        } else {
          current_scale_val = static_cast<ScaleType>(0.0f);  // Use zero padding
        }
      }

      // 3. Dequantize Weight using Shared Memory and Cached Params
      // 3a. Get packed weight from shared memory tile
      //    Use tk for row, block_packed_col_offset for the column this thread
      //    reads
      const int32_t packed_qwt_tile = sh_qwt[tk][block_packed_col_offset];

      // 3b. Unpack 4-bit weight and zero using pre-calculated physical index
      // and cached zero
      const uint32_t q_w =
          (static_cast<uint32_t>(packed_qwt_tile) >> bit_shift) & 0x0F;
      const uint32_t q_z =
          (static_cast<uint32_t>(current_packed_zos) >> bit_shift) & 0x0F;

      // 3c. Compute dequantized weight using cached scale
      const float scale_float = static_cast<float>(current_scale_val);
      const float dequant_w =
          (static_cast<float>(q_w) - static_cast<float>(q_z)) * scale_float;

      // 4. Accumulate product using FMA
      acc = __fmaf_rn(inp_val, dequant_w, acc);

    }  // End loop over tk (TILE_K)

    // Implicit __syncthreads() before the next k_base iteration starts (at the
    // beginning of the loop)

  }  // End loop over k_base (Tiles)

  // --- Add Bias (if provided) ---
  // Check 'n' boundary again before potentially accessing bias and writing
  // output
  if (n < N) {
    if (bias != nullptr) {
      acc += static_cast<float>(bias[n]);
    }

    // --- Store Result to Global Memory ---
    out[m * N + n] = static_cast<T>(acc);
  }

}  // Optimized kernel function end

//----------------------------------------------------------------------------
// 封装函数 (调用优化的计算内核) - Validation logic from original v12
//----------------------------------------------------------------------------
template <typename T>
void matmul_quantized(
    const Tensor<T> &input,          // 输入张量 [M, K]
    const Tensor<int32_t> &qweight,  // 量化权重张量 [K, N / PACK_FACTOR]
    const Tensor<float>
        &scales_input,  // 缩放因子张量 [NumGroups, N] (封装层假设为 float)
    const Tensor<int32_t>
        &zeros_input,       // 量化零点张量 [NumGroups, N / PACK_FACTOR]
    int group_size,         // 分组大小
    Tensor<T> *output,      // 输出张量 [M, N]
    cudaStream_t stream,    // CUDA 流
    const Tensor<T> *bias)  // 可选的偏置张量 [N] (可以为nullptr)
{
  // --- 输入验证、维度推导、形状检查 (Copied from user's v12) ---
  const int M = static_cast<int>(input.sizes()[0]);
  const int K = static_cast<int>(input.sizes()[1]);
  int N = 0;
  // NOTE: Using the potentially fragile N inference from user's v12 as
  // requested
  if (scales_input.sizes().size() == 2)
    N = static_cast<int>(scales_input.sizes()[1]);
  else if (output->sizes().size() ==
           2)  // This might be problematic if output is not pre-sized
    N = static_cast<int>(output->sizes()[1]);
  else if (qweight.sizes().size() == 2)
    N = static_cast<int>(qweight.sizes()[1]) * PACK_FACTOR;
  else
    // Keep the original error throwing behavior if N cannot be inferred
    throw std::runtime_error("无法确定维度 N");

  // Validation checks from user's v12
  if (K <= 0 || N <= 0 || M <= 0 || group_size <= 0 || K % group_size != 0) {
    // Basic checks
    throw std::runtime_error(
        "Invalid dimensions or group_size. M=" + std::to_string(M) +
        ", K=" + std::to_string(K) + ", N=" + std::to_string(N) +
        ", group_size=" + std::to_string(group_size));
  }
  const int NumGroups = K / group_size;

  // Shape consistency checks from user's v12
  if (input.sizes().size() != 2 || input.sizes()[0] != M ||
      input.sizes()[1] != K)
    throw std::runtime_error("Input shape mismatch. Expected (" +
                             std::to_string(M) + ", " + std::to_string(K) +
                             "), Got " + format_sizes(input.sizes()));
  if (qweight.sizes().size() != 2 || qweight.sizes()[0] != K ||
      qweight.sizes()[1] != (N / PACK_FACTOR))
    throw std::runtime_error("QWeight shape mismatch. Expected (" +
                             std::to_string(K) + ", " +
                             std::to_string(N / PACK_FACTOR) + "), Got " +
                             format_sizes(qweight.sizes()));
  if (scales_input.sizes().size() != 2 ||
      scales_input.sizes()[0] != NumGroups || scales_input.sizes()[1] != N)
    throw std::runtime_error(
        "Scales shape mismatch. Expected (" + std::to_string(NumGroups) + ", " +
        std::to_string(N) + "), Got " + format_sizes(scales_input.sizes()));
  if (zeros_input.sizes().size() != 2 || zeros_input.sizes()[0] != NumGroups ||
      zeros_input.sizes()[1] != (N / PACK_FACTOR))
    throw std::runtime_error("Zeros shape mismatch. Expected (" +
                             std::to_string(NumGroups) + ", " +
                             std::to_string(N / PACK_FACTOR) + "), Got " +
                             format_sizes(zeros_input.sizes()));
  // Output check should ideally ensure output *can be* M,N, but v12 didn't
  // force resize here. Let's check if it *is* M, N if preallocated, matching
  // v12's check style.

  if (bias && (bias->sizes().size() != 1 || bias->sizes()[0] != N))
    throw std::runtime_error("Bias shape mismatch. Expected (" +
                             std::to_string(N) + "), Got " +
                             format_sizes(bias->sizes()));

  // --- Kernel Launch Configuration ---
  // Define tile/block dimensions (can be tuned)
  constexpr int TILE_K_VAL = 32;  // K-dimension tile size
  constexpr int BLOCK_N_VAL =
      256;  // Threads per block (must be multiple of warp size & PACK_FACTOR)
  constexpr int BLOCK_M_VAL = 1;  // Rows processed per block

  // Ensure BLOCK_N is valid (compile-time check)
  static_assert(BLOCK_N_VAL % PACK_FACTOR == 0,
                "BLOCK_N must be a multiple of PACK_FACTOR (8)");
  static_assert(BLOCK_N_VAL % 32 == 0,
                "BLOCK_N should be a multiple of warp size (32)");

  const dim3 block_size(
      BLOCK_N_VAL);  // 1D block: BLOCK_N_VAL threads along N dimension
  // Grid: Cover N columns using blocks of size BLOCK_N_VAL, cover M rows using
  // Y dimension
  const dim3 grid_size((N + BLOCK_N_VAL - 1) / BLOCK_N_VAL, M);

#ifdef DEBUG_AWQ
  std::cout << "\nLaunching AWQ Optimized Kernel (v14 based on v12)..."
            << std::endl;
  std::cout << "  Grid Size: (" << grid_size.x << ", " << grid_size.y << ")"
            << std::endl;
  std::cout << "  Block Size: (" << block_size.x << ")" << std::endl;
  std::cout << "  Tile K: " << TILE_K_VAL << std::endl;
  std::cout << "  M=" << M << ", K=" << K << ", N=" << N
            << ", group_size=" << group_size << std::endl;
  debug_print_tensor(input, "Input");
  debug_print_tensor(qweight, "QWeight");
  debug_print_tensor(scales_input, "Scales");
  debug_print_tensor(zeros_input, "Zeros");
  if (bias) debug_print_tensor(*bias, "Bias");
#endif

  // --- Select Scale Type ---
  // Assuming scales are float as per interface
  using ScaleType = float;

  // --- Launch the Optimized CUDA Kernel ---
  matmul_awq_kernel_optimized<T, ScaleType, TILE_K_VAL, BLOCK_N_VAL,
                              BLOCK_M_VAL>
      <<<grid_size, block_size, 0, stream>>>(
          input.data_ptr(), qweight.data_ptr(), scales_input.data_ptr(),
          zeros_input.data_ptr(), output->data_ptr(), M, K, N, group_size,
          bias ? bias->data_ptr() : nullptr);

  // --- Error Checking ---
  cudaError_t launch_err = cudaGetLastError();
  if (launch_err != cudaSuccess) {
    std::cerr << "CUDA kernel LAUNCH error (matmul_awq_kernel_optimized): "
              << cudaGetErrorString(launch_err) << std::endl;
    throw std::runtime_error("CUDA kernel launch error");
  }

#ifdef DEBUG_AWQ
  // Optional: Add sync and print output for debugging
  cudaStreamSynchronize(stream);
  cudaError_t sync_err = cudaGetLastError();
  if (sync_err != cudaSuccess) {
    std::cerr << "CUDA sync error after kernel: "
              << cudaGetErrorString(sync_err) << std::endl;
  } else {
    debug_print_tensor(*output, "Output", 20);  // Print more elements of output
  }
#endif
}

// --- 显式模板实例化 ---
// (保持与之前版本相同)
template void matmul_quantized<float>(const Tensor<float> &,
                                      const Tensor<int32_t> &,
                                      const Tensor<float> &,
                                      const Tensor<int32_t> &, int,
                                      Tensor<float> *, cudaStream_t,
                                      const Tensor<float> *);
template void matmul_quantized<__nv_bfloat16>(
    const Tensor<__nv_bfloat16> &, const Tensor<int32_t> &,
    const Tensor<float> &, const Tensor<int32_t> &, int,
    Tensor<__nv_bfloat16> *, cudaStream_t, const Tensor<__nv_bfloat16> *);
template void matmul_quantized<__half>(const Tensor<__half> &,
                                       const Tensor<int32_t> &,
                                       const Tensor<float> &,
                                       const Tensor<int32_t> &, int,
                                       Tensor<__half> *, cudaStream_t,
                                       const Tensor<__half> *);

}  // namespace cuda_OP