#include <cmath>
#include <cstdio>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>       // For std::to_string
#include <type_traits>  // For std::is_same
#include <vector>

#include "cudaOP.cuh"  // Include the header we just defined

// ================================================================
// Helper: Warp Reduction
// ================================================================
template <int warp_size>
static __device__ inline float warp_reduce_sum(float val) {
#pragma unroll
  for (int mask = warp_size / 2; mask > 0; mask /= 2) {
    val += __shfl_xor_sync(0xffffffff, val, mask, warp_size);
  }
  return val;
}

namespace cuda_OP {

// ================================================================
// Kernels
// ================================================================

// ----------------------------------------------------------------
// Kernel 1: Matrix-Vector Multiplication (Original, bias separate)
// Inputs and Output all have type T_weight (float or nv_bfloat16)
// ----------------------------------------------------------------
template <typename T_weight, int block_size>
static __global__ void matmul_vector_kernel(
    const T_weight* __restrict__ x, const T_weight* __restrict__ y,
    T_weight* __restrict__ dst, const int64_t ncols2, const int64_t stride_row,
    const int64_t channel_ratio, const int64_t stride_channel_x,
    const int64_t stride_channel_y, const int64_t stride_channel_dst,
    const int64_t sample_ratio, const int64_t stride_sample_x,
    const int64_t stride_sample_y, const int64_t stride_sample_dst) {
  const int64_t row = blockIdx.x;
  const int tid = threadIdx.x;
  constexpr int warp_size = 32;

  x += row * stride_row;

  extern __shared__ char data_smem_mv[];  // Use unique name for shared mem
  float* buf_iw = (float*)data_smem_mv;
  if (block_size > warp_size) {
    int num_warps_in_block = (block_size + warp_size - 1) / warp_size;
    if (tid < num_warps_in_block) {
      buf_iw[tid] = 0.0f;
    }
    __syncthreads();
  }

  float sumf = 0.0f;
  for (int64_t col2 = tid; col2 < ncols2; col2 += block_size) {
    float tmpx_x_float, tmpx_y_float;
    float tmpy_x_float, tmpy_y_float;

    if constexpr (std::is_same<T_weight, float>::value) {
      const float2* x2 = (const float2*)x;
      const float2* y2 = (const float2*)y;
      const float2 tmpx = x2[col2];
      const float2 tmpy = y2[col2];
      tmpx_x_float = tmpx.x;
      tmpx_y_float = tmpx.y;
      tmpy_x_float = tmpy.x;
      tmpy_y_float = tmpy.y;
    } else if constexpr (std::is_same<T_weight, nv_bfloat16>::value) {
      const nv_bfloat162* x2 = (const nv_bfloat162*)x;
      const nv_bfloat162* y2 = (const nv_bfloat162*)y;
      const nv_bfloat162 tmpx = x2[col2];
      const nv_bfloat162 tmpy = y2[col2];
      tmpx_x_float = __bfloat162float(tmpx.x);
      tmpx_y_float = __bfloat162float(tmpx.y);
      tmpy_x_float = __bfloat162float(tmpy.x);
      tmpy_y_float = __bfloat162float(tmpy.y);
    }
    sumf += tmpx_x_float * tmpy_x_float;
    sumf += tmpx_y_float * tmpy_y_float;
  }

  sumf = warp_reduce_sum<warp_size>(sumf);
  if (block_size > warp_size) {
    if ((tid % warp_size) == 0) {
      buf_iw[tid / warp_size] = sumf;
    }
    __syncthreads();
    int num_warps = (block_size + warp_size - 1) / warp_size;
    if (tid < warp_size) {
      sumf = (tid < num_warps) ? buf_iw[tid] : 0.0f;
    } else {
      sumf = 0.0f;
    }
    if (tid < warp_size) {
      sumf = warp_reduce_sum<warp_size>(sumf);
    }
  }

  if (tid == 0) {
    if constexpr (std::is_same<T_weight, float>::value) {
      dst[row] = sumf;
    } else if constexpr (std::is_same<T_weight, nv_bfloat16>::value) {
      dst[row] = __float2bfloat16_rn(sumf);
    }
  }
}

// ----------------------------------------------------------------
// Kernel 2: Elementwise Bias Addition (Original, separate kernel)
// ----------------------------------------------------------------
template <typename T_weight>
static __global__ void add_vector_bias_kernel(T_weight* y_out,
                                              const T_weight* bias_vec, int P) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < P) {
    T_weight y_val = y_out[j];
    T_weight bias_val = bias_vec[j];
    if constexpr (std::is_same<T_weight, float>::value) {
      y_out[j] = y_val + bias_val;
    } else if constexpr (std::is_same<T_weight, nv_bfloat16>::value) {
      float y_f = __bfloat162float(y_val);
      float b_f = __bfloat162float(bias_val);
      y_out[j] = __float2bfloat16_rn(y_f + b_f);
    }
  }
}

// ----------------------------------------------------------------
// Kernel 3: Matrix-Vector Multiplication with Fused Bias Addition
// (New optimized kernel)
// ----------------------------------------------------------------
template <typename T_weight, int block_size>
static __global__ void
matmul_vector_kernel_optimized_fused_bias(  // Renamed for clarity
    const T_weight* __restrict__ x,         // Matrix M (PxN)
    const T_weight* __restrict__ y,         // Vector v (N)
    T_weight* __restrict__ dst,             // Vector y_out (P)
    const T_weight* __restrict__ bias_vec,  // Bias vector (P) - Can be nullptr
                                            // <<-- ADDED BIAS PARAM
    const int64_t ncols2,                   // N / 2 (e.g., 768 for N=1536)
    const int64_t stride_row,               // N (Row stride of M)
    // --- Unused interface parameters (kept for signature compatibility) ---
    const int64_t channel_ratio, const int64_t stride_channel_x,
    const int64_t stride_channel_y, const int64_t stride_channel_dst,
    const int64_t sample_ratio, const int64_t stride_sample_x,
    const int64_t stride_sample_y, const int64_t stride_sample_dst) {
  // --- Compile-time Type Determination for Vec2 ---
  using TVec2 =
      typename std::conditional<std::is_same<T_weight, float>::value, float2,
                                nv_bfloat162  // Assume if not float, it's
                                              // bfloat16
                                >::type;

  // --- Thread/Block Indexing ---
  const int64_t row = blockIdx.x;  // Output element index (0 to P-1)
  const int tid = threadIdx.x;     // Thread ID within block (0 to block_size-1)
  constexpr int warp_size = 32;

  // --- Shared Memory Declaration ---
  // Static shared memory for caching vector y. Requires N to be known/fixed.
  constexpr int N_COLS2_EXPECTED = 1536 / 2;  // Assuming N=1536 -> ncols2=768
  __shared__ TVec2 y_s_cache[N_COLS2_EXPECTED];

  // Shared memory for reduction buffer (dynamic size okay via extern)
  extern __shared__ char data_smem_reduction[];
  float* buf_iw = (float*)data_smem_reduction;

  // --- Load Vector y into Shared Memory ---
  const TVec2* y_vec_ptr = reinterpret_cast<const TVec2*>(y);
  for (int i = tid; i < ncols2; i += block_size) {
    if (i < N_COLS2_EXPECTED) {  // Check against cache size
      y_s_cache[i] = y_vec_ptr[i];
    }
    // Add error handling or assertion if ncols2 > N_COLS2_EXPECTED
  }

  // --- Initialize Reduction Buffer ---
  if (block_size > warp_size) {
    int num_warps_in_block = (block_size + warp_size - 1) / warp_size;
    if (tid < num_warps_in_block) {
      buf_iw[tid] = 0.0f;
    }
  }

  // --- Synchronize ---
  __syncthreads();  // Wait for shared memory load and init

  // --- Dot Product Calculation: dot(M[row, :], v[:]) ---
  float sumf = 0.0f;
  const T_weight* x_row_start = x + row * stride_row;

  for (int64_t col2 = tid; col2 < ncols2; col2 += block_size) {
    if (col2 < N_COLS2_EXPECTED) {  // Check against cache size
      const TVec2 tmpx_vec = reinterpret_cast<const TVec2*>(x_row_start)[col2];
      const TVec2 tmpy_vec = y_s_cache[col2];  // Read y from shared memory

      float tmpx_x_float, tmpx_y_float, tmpy_x_float, tmpy_y_float;
      if constexpr (std::is_same<T_weight, float>::value) {
        tmpx_x_float = tmpx_vec.x;
        tmpx_y_float = tmpx_vec.y;
        tmpy_x_float = tmpy_vec.x;
        tmpy_y_float = tmpy_vec.y;
      } else if constexpr (std::is_same<T_weight, nv_bfloat16>::value) {
        tmpx_x_float = __bfloat162float(tmpx_vec.x);
        tmpx_y_float = __bfloat162float(tmpx_vec.y);
        tmpy_x_float = __bfloat162float(tmpy_vec.x);
        tmpy_y_float = __bfloat162float(tmpy_vec.y);
      }
      sumf += tmpx_x_float * tmpy_x_float;
      sumf += tmpx_y_float * tmpy_y_float;
    }
  }

  // --- Reduction Phase (same as before) ---
  sumf = warp_reduce_sum<warp_size>(sumf);
  if (block_size > warp_size) {
    if ((tid % warp_size) == 0) {
      buf_iw[tid / warp_size] = sumf;
    }
    __syncthreads();
    int num_warps = (block_size + warp_size - 1) / warp_size;
    if (tid < warp_size) {
      sumf = (tid < num_warps) ? buf_iw[tid] : 0.0f;
    } else {
      sumf = 0.0f;
    }
    if (tid < warp_size) {
      sumf = warp_reduce_sum<warp_size>(sumf);
    }
  }

  // --- Write Output with Fused Bias ---
  if (tid == 0) {
    float final_val = sumf;  // Start with the matmul result

    // Add bias if bias_vec pointer is not null
    if (bias_vec != nullptr) {
      T_weight bias_val = bias_vec[row];  // Read bias for this row
      if constexpr (std::is_same<T_weight, float>::value) {
        final_val += bias_val;  // Add float bias
      } else if constexpr (std::is_same<T_weight, nv_bfloat16>::value) {
        final_val +=
            __bfloat162float(bias_val);  // Convert bias to float and add
      }
      // Add other types if needed
    }

    // Convert final result back to T_weight and write
    if constexpr (std::is_same<T_weight, float>::value) {
      dst[row] = final_val;
    } else if constexpr (std::is_same<T_weight, nv_bfloat16>::value) {
      dst[row] = __float2bfloat16_rn(final_val);
    }
    // Add other types if needed
  }
}
// ----------------------------------------------------------------
// C++ Wrapper Function Definition (Using Fused Kernel by Default)
// ----------------------------------------------------------------
template <typename T_weight>
void decode_qkv_matmul(const Tensor<T_weight>& weight,
                       const Tensor<T_weight>& qkv_decode,
                       Tensor<T_weight>* out, cudaStream_t stream,
                       const Tensor<T_weight>* bias) {
  // --- 1. Type and Shape Checks ---
  static_assert(std::is_same<T_weight, nv_bfloat16>::value ||
                    std::is_same<T_weight, float>::value,
                "decode_qkv_matmul: T_weight must be nv_bfloat16 or float.");

  const std::vector<size_t>& M_shape = weight.sizes();
  const std::vector<size_t>& v_shape = qkv_decode.sizes();
  const std::vector<size_t>& y_shape = out->sizes();

  // Basic dimension count check and extract P, N
  if (!(M_shape.size() == 2 &&
        (v_shape.size() == 1 ||
         (v_shape.size() == 2 && v_shape[0] == 1 && v_shape[1] > 0) ||
         (v_shape.size() == 2 && v_shape[1] == 1 &&
          v_shape[0] > 0)) &&  // Allow [N] or [1,N] or [N,1]
        (y_shape.size() == 1 ||
         (y_shape.size() == 2 && y_shape[0] == 1 && y_shape[1] > 0) ||
         (y_shape.size() == 2 && y_shape[1] == 1 &&
          y_shape[0] > 0)))) {  // Allow [P] or [1,P] or [P,1]
    throw std::runtime_error("decode_qkv_matmul: Invalid tensor dimensions.");
  }

  // Assume weight is [P, N] row-major
  size_t P = M_shape[1];
  size_t N = M_shape[0];

  // Infer N from qkv_decode (handle [N], [1,N], [N,1])
  size_t N_check_v = 0;
  if (v_shape.size() == 1)
    N_check_v = v_shape[0];
  else if (v_shape.size() == 2)
    N_check_v = (v_shape[0] == 1)
                    ? v_shape[1]
                    : v_shape[0];  // If [1,N] take N, if [N,1] take N

  // Infer P from out (handle [P], [1,P], [P,1])
  size_t P_check_y = 0;
  if (y_shape.size() == 1)
    P_check_y = y_shape[0];
  else if (y_shape.size() == 2)
    P_check_y = (y_shape[0] == 1)
                    ? y_shape[1]
                    : y_shape[0];  // If [1,P] take P, if [P,1] take P

  if (N != N_check_v)
    throw std::runtime_error("decode_qkv_matmul: Inner dimension N mismatch.");
  if (P != P_check_y)
    throw std::runtime_error("decode_qkv_matmul: Output dimension P mismatch.");
  if (N == 0 || P == 0)
    throw std::runtime_error(
        "decode_qkv_matmul: Dimensions P and N cannot be zero.");  // Check for
                                                                   // zero
                                                                   // dimensions
  if (N % 2 != 0)
    throw std::runtime_error(
        "decode_qkv_matmul: Inner dimension N must be even.");
  if (out->data_ptr() == nullptr || out->numel() < P)
    throw std::runtime_error(
        "decode_qkv_matmul: Output tensor 'out' is not allocated or too "
        "small.");

  // Bias Shape Check
  if (bias != nullptr) {
    const std::vector<size_t>& bias_shape = bias->sizes();
    size_t bias_len = bias->numel();  // Get total elements in bias
    if (bias_len != P) {              // Check if number of elements matches P
      throw std::runtime_error(
          "decode_qkv_matmul: Bias tensor number of elements (" +
          std::to_string(bias_len) + ") does not match output dimension P (" +
          std::to_string(P) + ").");
    }
    if (bias->data_ptr() == nullptr) {
      throw std::runtime_error(
          "decode_qkv_matmul: Bias tensor is provided but its data pointer is "
          "null.");
    }
  }

  // --- 2. Extract Data Pointers ---
  const T_weight* M_ptr = weight.data_ptr();
  const T_weight* v_ptr = qkv_decode.data_ptr();
  T_weight* y_ptr = out->data_ptr();
  const T_weight* bias_ptr = (bias != nullptr) ? bias->data_ptr() : nullptr;

  // --- 3. Kernel Launch ---
  // We will directly use the fused kernel.
  constexpr int block_size = 256;
  dim3 gridDim(P, 1, 1);
  dim3 blockDim(block_size, 1, 1);
  size_t shared_mem_size = 0;
  constexpr int warp_size = 32;
  if (block_size > warp_size) {
    int num_warps = (block_size + warp_size - 1) / warp_size;
    shared_mem_size = num_warps * sizeof(float);
  }
  const int64_t ncols2_arg = N / 2;
  const int64_t stride_row_arg = N;  // Assuming row-major for weight matrix
  const int64_t channel_ratio_arg = 1, stride_channel_x_arg = 0,
                stride_channel_y_arg = 0, stride_channel_dst_arg = 0;
  const int64_t sample_ratio_arg = 1, stride_sample_x_arg = 0,
                stride_sample_y_arg = 0, stride_sample_dst_arg = 0;

  // Launch the FUSED kernel

  if (bias != nullptr)
    matmul_vector_kernel_optimized_fused_bias<T_weight, block_size>
        <<<gridDim, blockDim, shared_mem_size, stream>>>(
            M_ptr, v_ptr, y_ptr,
            bias_ptr,  // Pass bias pointer (can be nullptr)
            ncols2_arg, stride_row_arg, channel_ratio_arg, stride_channel_x_arg,
            stride_channel_y_arg, stride_channel_dst_arg, sample_ratio_arg,
            stride_sample_x_arg, stride_sample_y_arg, stride_sample_dst_arg);
  else {
    matmul_vector_kernel<T_weight, block_size>
        <<<gridDim, blockDim, shared_mem_size, stream>>>(
            M_ptr, v_ptr, y_ptr, ncols2_arg, stride_row_arg, channel_ratio_arg,
            stride_channel_x_arg, stride_channel_y_arg, stride_channel_dst_arg,
            sample_ratio_arg, stride_sample_x_arg, stride_sample_y_arg,
            stride_sample_dst_arg);
  }

  // ---- End of separate kernel approach code ---- */

  // --- 5. Check for CUDA Errors ---
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    cudaError_t clearErr = cudaGetLastError();
    (void)clearErr;
    throw std::runtime_error(
        "CUDA kernel launch failed in decode_qkv_matmul (using fused kernel "
        "approach): " +
        std::string(cudaGetErrorString(err)));
  }
}

// --- Template Instantiations (Define in .cu file) ---
template void decode_qkv_matmul<float>(const Tensor<float>& weight,
                                       const Tensor<float>& qkv_decode,
                                       Tensor<float>* out, cudaStream_t stream,
                                       const Tensor<float>* bias);

template void decode_qkv_matmul<nv_bfloat16>(
    const Tensor<nv_bfloat16>& weight, const Tensor<nv_bfloat16>& qkv_decode,
    Tensor<nv_bfloat16>* out, cudaStream_t stream,
    const Tensor<nv_bfloat16>* bias);

}  // namespace cuda_OP