#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>  // printf
#include <iostream>
#include <stdexcept>
#include <vector>

#include "cudaOP.cuh"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"

inline void checkCublasStatus(cublasStatus_t status, const char *file,
                              int line) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    char errorMsg[256];
    // Note: cublasGetErrorString is not a standard function.
    // Provide a basic message.
    snprintf(errorMsg, sizeof(errorMsg), "cuBLAS error %d at %s:%d",
             static_cast<int>(status), file, line);
    fprintf(stderr, "%s\n", errorMsg);
    throw std::runtime_error(errorMsg);
  }
}
#define CHECK_CUBLAS(call) checkCublasStatus(call, __FILE__, __LINE__)

namespace cuda_OP {
// 定义类型转换 traits（可扩展支持更多类型）
template <typename T>
struct to_cutlass_type {
  using type = T;
};

template <>
struct to_cutlass_type<__nv_bfloat16> {
  using type = cutlass::bfloat16_t;
};

template <typename ElementA, typename ElementB, typename ElementOutput,
          typename LayoutA, typename LayoutB, typename LayoutOutput,
          typename ElementAccumulator = float,
          typename ElementComputeEpilogue = ElementAccumulator,
          typename MMAOp = cutlass::arch::OpClassTensorOp,
          typename SmArch = cutlass::arch::Sm80,
          typename ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>,
          typename ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>,
          typename ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>,
          typename SwizzleThreadBlock =
              cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
          int NumStages = 2>
cutlass::Status run_cutlass_gemm_raw_templated(
    int m, int n, int k, ElementA const *d_a, ElementB const *d_b,
    ElementOutput const *d_bias, ElementOutput *d_d, cudaStream_t stream = 0,
    ElementComputeEpilogue alpha = ElementComputeEpilogue(1),
    int split_k_slices = 1) {
  // 使用 to_cutlass_type 对输入数据类型做转换
  using ElementA_t = typename to_cutlass_type<ElementA>::type;
  using ElementB_t = typename to_cutlass_type<ElementB>::type;
  using ElementOutput_t = typename to_cutlass_type<ElementOutput>::type;

  // 定义 epilogue 操作（注意用转换后的 ElementOutput_t）
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      ElementOutput_t, 128 / cutlass::sizeof_bits<ElementOutput_t>::value,
      ElementAccumulator, ElementComputeEpilogue,
      cutlass::epilogue::thread::ScaleType::NoBetaScaling>;
  // std::cout << "value: " << cutlass::sizeof_bits<ElementOutput_t>::value
  //           << std::endl;
  // 定义 GEMM 操作类型，使用转换后的类型
  using Gemm = cutlass::gemm::device::Gemm<
      ElementA_t, LayoutA, ElementB_t, LayoutB, ElementOutput_t, LayoutOutput,
      ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp,
      ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages, 8, 8>;

  // 构造问题尺寸
  cutlass::gemm::GemmCoord problem_size(m, n, k);

  cutlass::TensorRef<ElementA_t, LayoutA> ref_A(
      const_cast<ElementA_t *>(reinterpret_cast<const ElementA_t *>(d_a)),
      LayoutA(k));
  cutlass::TensorRef<ElementB_t, LayoutB> ref_B(
      const_cast<ElementB_t *>(reinterpret_cast<const ElementB_t *>(d_b)),
      LayoutB(n));
  cutlass::TensorRef<ElementOutput_t, LayoutOutput> ref_D(
      reinterpret_cast<ElementOutput_t *>(d_d), LayoutOutput(n));

  // 构造 Gemm kernel 参数。bias 同样进行类型转换
  typename Gemm::Arguments arguments{
      problem_size,  ref_A,
      ref_B,         {reinterpret_cast<const ElementOutput_t *>(d_bias), 0},
      ref_D,         {alpha},
      split_k_slices};

  // 查询 workspace 内存大小并分配
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // 实例化 GEMM 对象，并检查问题是否可实现
  Gemm gemm_op;
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);

  // 初始化 GEMM 操作
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);

  // 调用 CUTLASS GEMM 内核
  status = gemm_op(stream);
  CUTLASS_CHECK(status);

  return status;
}

// --------------------------------------------------
// --------------------------------------------------
template <typename T>
__global__ void matmul_kernel(const T *A, const T *B, T *C, int M, int K,
                              int N) {
  __shared__ T As[16][16];
  __shared__ T Bs[16][16];
  int row = blockIdx.y * 16 + threadIdx.y;
  int col = blockIdx.x * 16 + threadIdx.x;
  T sum = T(0);
  // 计算需要的 tile 数量
  int numTiles = (K + 16 - 1) / 16;
  for (int t = 0; t < numTiles; ++t) {
    int A_col = t * 16 + threadIdx.x;
    if (row < M && A_col < K) {
      As[threadIdx.y][threadIdx.x] = A[row * K + A_col];
    } else {
      As[threadIdx.y][threadIdx.x] = T(0);
    }
    int B_row = t * 16 + threadIdx.y;
    if (col < N && B_row < K) {
      Bs[threadIdx.y][threadIdx.x] = B[col * K + B_row];
    } else {
      Bs[threadIdx.y][threadIdx.x] = T(0);
    }
    __syncthreads();
    for (int k = 0; k < 16; ++k) {
      sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    }
    __syncthreads();
    // Use code with caution.
  }
  if (row < M && col < N) {
    C[row * N + col] = sum;
  }
}
template <typename T>
__global__ void add_bias_kernel(T *C, const T *bias, int M, int N, int ldc) {
  // 计算当前线程负责处理的全局行索引 (row) 和列索引 (col)
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  // 边界检查，确保线程在矩阵的有效范围内
  if (row < M && col < N) {
    // 计算 C 矩阵中元素的线性索引 (行主序)
    int c_index = row * ldc + col;

    // 执行加法: C[row][col] = C[row][col] + bias[col]
    C[c_index] = C[c_index] + bias[col];
  }
}
// --------------------------------------------------
// 带偏置的矩阵乘法kernel
// --------------------------------------------------
template <typename T>
__global__ void matmul_with_bias_kernel(const T *A, const T *B, const T *bias,
                                        T *C, int M, int K, int N) {
  __shared__ T As[16][16];
  __shared__ T Bs[16][16];
  int row = blockIdx.y * 16 + threadIdx.y;
  int col = blockIdx.x * 16 + threadIdx.x;
  T sum = T(0);
  // 计算需要的 tile 数量
  int numTiles = (K + 16 - 1) / 16;
  for (int t = 0; t < numTiles; ++t) {
    int A_col = t * 16 + threadIdx.x;
    if (row < M && A_col < K) {
      As[threadIdx.y][threadIdx.x] = A[row * K + A_col];
    } else {
      As[threadIdx.y][threadIdx.x] = T(0);
    }
    int B_row = t * 16 + threadIdx.y;
    if (col < N && B_row < K) {
      Bs[threadIdx.y][threadIdx.x] = B[col * K + B_row];
    } else {
      Bs[threadIdx.y][threadIdx.x] = T(0);
    }
    __syncthreads();
    for (int k = 0; k < 16; ++k) {
      sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    }
    __syncthreads();
    // Use code with caution.
  }
  if (row < M && col < N) {
    // 在结果中加上偏置
    C[row * N + col] = sum + bias[col];
  }
}

/**
 * @brief 基于模板的 cuBLAS GEMM 包装函数，输入和输出类型相同。
 *
 * 执行操作：C = alpha * op(A) * op(B) + beta * C
 * 输入矩阵 A、B 以及输出矩阵 C 的类型都由模板参数 InputType 决定
 * (必须是 float 或 nv_bfloat16)。
 * 计算通常在内部使用 FP32 精度执行（由 compute_type
 * 控制），以获得更好的精度和性能。 结果直接写入 d_C 指向的设备内存中。
 *
 * @tparam InputType   输入矩阵 A, B 和输出矩阵 C 的数据类型 (float 或
 * nv_bfloat16)。
 * @param handle        cuBLAS 库句柄。
 * @param transa        指定操作 op(A)：CUBLAS_OP_N 或 CUBLAS_OP_T。
 * @param transb        指定操作 op(B)：CUBLAS_OP_N 或 CUBLAS_OP_T。
 * @param m             矩阵 op(A) 和矩阵 C 的行数。
 * @param n             矩阵 op(B) 和矩阵 C 的列数。
 * @param k             矩阵 op(A) 的列数和矩阵 op(B) 的行数。
 * @param alpha         用于 op(A) * op(B) 的标量乘子 (指向 const float
 * 的主机指针)。
 * @param d_A           指向设备内存中矩阵 A 的指针 (const InputType*)。
 * @param lda           矩阵 A 的主维度。
 * @param d_B           指向设备内存中矩阵 B 的指针 (const InputType*)。
 * @param ldb           矩阵 B 的主维度。
 * @param beta          用于矩阵 C 的标量乘子 (指向 const float 的主机指针)。
 * @param d_C           指向设备内存中矩阵 C 的指针
 * (InputType*)。结果将写入此处，类型与输入匹配。
 * @param ldc           矩阵 C 的主维度。
 */
template <typename InputType>
void cublas_matmul_wrapper(cublasHandle_t handle, cublasOperation_t transa,
                           cublasOperation_t transb, int m, int n, int k,
                           const float *alpha,  // 改回 const float*
                           const InputType *d_A, int lda, const InputType *d_B,
                           int ldb,
                           const float *beta,  // 改回 const float*
                           InputType *d_C,  // *** 输出类型为 InputType* ***
                           int ldc) {
  // printf("--- 进入 cublas_matmul_wrapper ---\n");
  // printf("模板类型 InputType: %s\n", std::is_same_v<InputType, float> ?
  // "float"
  //                                    : std::is_same_v<InputType, nv_bfloat16>
  //                                        ? "nv_bfloat16"
  //                                        : "未知");

  // // 打印输入参数
  // printf("cuBLAS Handle: %p\n", (void *)handle);
  // printf("transa: %d (N=%d, T=%d)\n", transa, CUBLAS_OP_N, CUBLAS_OP_T);
  // printf("transb: %d (N=%d, T=%d)\n", transb, CUBLAS_OP_N, CUBLAS_OP_T);
  // printf("m: %d, n: %d, k: %d\n", m, n, k);
  // printf("alpha: %f (来自地址 %p)\n", *alpha, (void *)alpha);
  // printf("d_A: %p\n", (void *)d_A);
  // printf("lda: %d\n", lda);
  // printf("d_B: %p\n", (void *)d_B);
  // printf("ldb: %d\n", ldb);
  // printf("beta: %f (来自地址 %p)\n", *beta, (void *)beta);
  // printf("d_C: %p\n", (void *)d_C);
  // printf("ldc: %d\n", ldc);
  // fflush(stdout);  // 确保在调用 cuBLAS 前打印出来

  cudaDataType_t cuda_data_type_A;
  cudaDataType_t cuda_data_type_B;
  cudaDataType_t cuda_data_type_C;

  cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F_FAST_TF32;
  // printf("启用 TF32 计算类型\n");

  // cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
  // printf("使用 FP32 计算类型\n");

  // 根据模板类型 InputType 确定 A, B, C 的 CUDA 数据类型
  if constexpr (std::is_same_v<InputType, nv_bfloat16>) {
    cuda_data_type_A = CUDA_R_16BF;
    cuda_data_type_B = CUDA_R_16BF;
    cuda_data_type_C = CUDA_R_16BF;
    // printf("数据类型: A=BF16, B=BF16, C=BF16\n");
  } else if constexpr (std::is_same_v<InputType, float>) {
    cuda_data_type_A = CUDA_R_32F;
    cuda_data_type_B = CUDA_R_32F;
    cuda_data_type_C = CUDA_R_32F;
    // printf("数据类型: A=FP32, B=FP32, C=FP32\n");
    // 可选: Ampere+ 可考虑 TF32 计算
    // compute_type = CUBLAS_COMPUTE_32F_FAST_TF32;
  } else {
    // 这个 static_assert 会在编译时检查，如果运行到这里说明模板参数类型不对
    // 但为了运行时更明确，可以加个错误打印
    // fprintf(stderr, "错误：不支持的 InputType!\n");
    static_assert(std::is_same_v<InputType, nv_bfloat16> ||
                      std::is_same_v<InputType, float>,
                  "cublas_matmul_wrapper 只支持 nv_bfloat16 和 float "
                  "输入/输出类型。");
    return;  // 或者抛出异常
  }

  // printf("计算类型 compute_type: %d (CUBLAS_COMPUTE_32F=%d)\n", compute_type,
  //        CUBLAS_COMPUTE_32F);

  // // --- 选择算法 ---
  cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;  // 让 cuBLAS 选择
  // printf("选择算法: CUBLAS_GEMM_DEFAULT (%d)\n", algo);
  // fflush(stdout);  // 再次确保打印

  // // --- 执行 GEMM 操作 ---
  // printf("即将调用 cublasGemmEx...\n");
  // fflush(stdout);

  cublasStatus_t status =
      cublasGemmEx(handle, transa, transb, m, n, k,
                   alpha,             // 标量 alpha (主机)
                   d_A,               // 矩阵 A (设备)
                   cuda_data_type_A,  // A 的类型
                   lda,
                   d_B,               // 矩阵 B (设备)
                   cuda_data_type_B,  // B 的类型
                   ldb,
                   beta,  // 标量 beta (主机)
                   d_C,   // 矩阵 C (设备) - 类型现在是 InputType*
                   cuda_data_type_C,  // C 的类型现在根据 InputType 确定
                   ldc,
                   compute_type,  // 内部计算精度 (推荐保持 FP32)
                   algo);

  // printf("cublasGemmEx 调用返回，状态码: %d\n", status);
  // fflush(stdout);

  // 使用 CHECK_CUBLAS 宏来检查返回状态
  CHECK_CUBLAS(status);

  // 可选：添加 CUDA 同步和错误检查，确保 GEMM 内核执行完成且没有异步错误
  // cudaError_t cuda_err = cudaDeviceSynchronize();
  // if (cuda_err != cudaSuccess) {
  //   fprintf(stderr, "CUDA error after cublasGemmEx sync: %s\n",
  //   cudaGetErrorString(cuda_err));
  // } else {
  //    printf("cudaDeviceSynchronize 成功\n");
  // }
  // fflush(stdout);

  // printf("--- 退出 cublas_matmul_wrapper ---\n");
  // fflush(stdout);  // 确保退出信息也打印出来
}
// --------------------------------------------------
// --------------------------------------------------
template <typename T>
void matmul(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> *C,
            cudaStream_t stream, const Tensor<T> *bias, int use_) {
  // 如果选择使用 cublas 计算，则调用 cublas 的包装函数接口

  const std::vector<size_t> &A_shape = A.sizes();
  const std::vector<size_t> &B_shape = B.sizes();

  // A: [M, K], B: [N, K]（保证 A 的第二维与 B 的第二维一致）
  size_t M = A_shape[0];
  size_t K = A_shape[1];
  size_t N = B_shape[1];

  if (bias == nullptr && use_ == 2) {
    use_ = 1;
  }

  if (use_ == 2) {
    cutlass::Status status = run_cutlass_gemm_raw_templated<
        T,                             // ElementA
        T,                             // ElementB
        T,                             // ElementOutput
        cutlass::layout::RowMajor,     // LayoutA
        cutlass::layout::ColumnMajor,  // LayoutB
        cutlass::layout::RowMajor      // LayoutOutput
        >(M, N, K, A.data_ptr(), B.data_ptr(), bias->data_ptr(), C->data_ptr(),
          stream);

  } else if (use_ == 1) {
    static cublasHandle_t handle = nullptr;
    // 使用静态标志和互斥锁确保线程安全的单次初始化
    static std::once_flag init_flag;
    static std::mutex handle_mutex;  // 保护对 handle 的并发使用 (如果需要)

    // 确保 cublasCreate 只被调用一次，且线程安全
    std::call_once(init_flag, [&]() {
      std::lock_guard<std::mutex> lock(handle_mutex);  // 锁定以进行创建
      // printf("--- Initializing static cublasHandle ---\n"); // 调试信息
      cublasStatus_t status = cublasCreate(&handle);
      if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "FATAL ERROR: cublasCreate failed in static init: %d\n",
                status);
        // 在这里可能需要做更健壮的错误处理，比如抛出异常或终止程序
        handle = nullptr;  // 确保句柄无效
      } else {
        // 可选：如果需要关联特定流，可以在这里设置
        // cudaStream_t stream = nullptr; // 或获取一个全局流
        // cublasSetStream(handle, stream);
        // printf("--- Static cublasHandle initialized: %p ---\n",
        // (void*)handle); // 调试信息

        // 注册程序退出时调用的清理函数
        std::atexit([]() {
          std::lock_guard<std::mutex> lock(handle_mutex);  // 同样锁定以进行销毁
          if (handle != nullptr) {
            // printf("--- Destroying static cublasHandle: %p ---\n",
            // (void*)handle); // 调试信息
            cublasDestroy(handle);
            handle = nullptr;  // 标记为已销毁
          }
        });
      }
    });

    // 检查初始化是否成功 (如果在 call_once 中处理失败)
    if (handle == nullptr) {
      fprintf(stderr, "Error: cuBLAS handle was not initialized correctly.\n");
      return;  // 或者抛出异常
    }
    // 原始数据（均按行主序存储）：
    // A: M×K, 每行有 K 个元素  -> lda = K
    // B: N×K, 每行有 K 个元素  -> ldb = K
    // C: M×N, 每行有 N 个元素  -> ldc = N
    int lda = K;  // A 每行有 K 个元素
    int ldb = K;  // B 每行有 K 个元素
    int ldc = N;  // C 每行有 N 个元素
    const float alpha = 1.0f;
    const float beta = 0.0f;
    {  // 引入作用域方便 lock_guard 管理
      std::lock_guard<std::mutex> lock(handle_mutex);
      // 将 cuBLAS 操作与传入的 stream 关联
      CHECK_CUBLAS(
          cublasSetStream(handle, stream));  // 确保 handle 使用正确的流

      // 目标计算： C = A * B^T
      // 利用转换： C^T = B * A^T
      // GEMM 调用参数：
      //    m = N, n = M, k = K
      // 同时对 A 和 B 使用转置操作：
      cublas_matmul_wrapper<T>(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                               int(N),  // m = N
                               int(M),  // n = M
                               int(K),  // k = K
                               &alpha,
                               B.data_ptr(),  // 原始 B
                               ldb,           // ldb = K
                               A.data_ptr(),  // 原始 A
                               lda,           // lda = K
                               &beta,
                               C->data_ptr(),  // 输出 C
                               ldc);           // ldc = N
      // cudaStreamSynchronize(stream);
    }
    // std::cout << "cublas_matmul_wrapper<T> 调用成功" << std::endl;
    if (bias != nullptr) {
      // printf("Launching add_bias_kernel:\n");
      // printf("  C ptr: %p\n", (void *)C->data_ptr());
      // printf("  bias ptr: %p\n", (void *)bias->data_ptr());
      // printf("  M: %d\n", static_cast<int>(M));
      // printf("  N: %d\n", static_cast<int>(N));
      // printf("  ldc: %d\n", ldc);
      // fflush(stdout);  // 确保打印出来

      // --- 检查指针是否为 NULL ---
      // if (C->data_ptr() == nullptr || bias->data_ptr() == nullptr) {
      //   fprintf(stderr, "错误: C 或 bias 指针为 NULL!\n");
      //   // 可能需要在这里中断或返回错误
      // }
      // // --- 检查 ldc ---
      // if (ldc < static_cast<int>(N)) {
      //   fprintf(stderr, "错误: ldc (%d) 小于 N (%d)!\n", ldc,
      //           static_cast<int>(N));
      //   // 可能需要在这里中断或返回错误
      // }
      // // --- 检查维度是否合理 ---
      // if (static_cast<int>(M) <= 0 || static_cast<int>(N) <= 0) {
      //   fprintf(stderr, "错误: M (%d) 或 N (%d) 无效!\n",
      //   static_cast<int>(M),
      //           static_cast<int>(N));
      //   // 可能需要在这里中断或返回错误
      // }

      dim3 blockDim(16, 16);
      // 计算网格大小，确保覆盖所有元素
      dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                   (M + blockDim.y - 1) / blockDim.y);
      add_bias_kernel<T><<<gridDim, blockDim, 0, stream>>>(
          C->data_ptr(), bias->data_ptr(), static_cast<int>(M),
          static_cast<int>(N), ldc);
    }
    // std::cout << "add_bias_kernel 调用成功" << std::endl;
    return;
  }

  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

  if (bias == nullptr) {
    // 使用无偏置版本的kernel
    matmul_kernel<T><<<numBlocks, threadsPerBlock, 0, stream>>>(
        A.data_ptr(), B.data_ptr(), C->data_ptr(), M, K, N);
  } else {
    // 检查偏置形状
    const std::vector<size_t> &bias_shape = bias->sizes();
    if (bias_shape.size() != 1) {
      throw std::runtime_error("Bias must be a 1D tensor");
    }
    if (bias_shape[0] != N) {
      throw std::runtime_error("Bias size must match output column dimension");
    }
    // 使用带偏置版本的kernel
    matmul_with_bias_kernel<T><<<numBlocks, threadsPerBlock, 0, stream>>>(
        A.data_ptr(), B.data_ptr(), bias->data_ptr(), C->data_ptr(), M, K, N);
    // Use code with caution.
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error("CUDA kernel launch failed: " +
                             std::string(cudaGetErrorString(err)));
  }

  return;
}

template void matmul<float>(const Tensor<float> &, const Tensor<float> &,
                            Tensor<float> *, cudaStream_t,
                            const Tensor<float> *, int);
template void matmul<__nv_bfloat16>(const Tensor<__nv_bfloat16> &,
                                    const Tensor<__nv_bfloat16> &,
                                    Tensor<__nv_bfloat16> *, cudaStream_t,
                                    const Tensor<__nv_bfloat16> *, int);

}  // namespace cuda_OP
