#pragma once
#include <cuda_runtime.h>

// C风格的枚举，用于在运行时安全地传递数据类型信息
typedef enum { MY_CUTLASS_DTYPE_BF16, MY_CUTLASS_DTYPE_FLOAT32 } my_cutlass_dtype_t;

// C风格的返回状态码，这是我们对外暴露的稳定契约
typedef enum {
    MY_CUTLASS_STATUS_SUCCESS = 0,
    MY_CUTLASS_STATUS_ERROR_INVALID_PROBLEM,
    MY_CUTLASS_STATUS_ERROR_NOT_SUPPORTED,
    MY_CUTLASS_STATUS_ERROR_INTERNAL
} my_cutlass_status_t;

#ifdef __cplusplus
extern "C" {
#endif

// 唯一的、提供给外部调用的 CUTLASS GEMM C 语言接口
// [最终修正] 返回值是且永远是 my_cutlass_status_t
void cutlass_gemm_c_api(int m, int n, int k, my_cutlass_dtype_t dtype, const void* ptr_a, const void* ptr_b,
                        const void* ptr_bias, void* ptr_d, cudaStream_t stream);

// 辅助函数：将状态码转换为字符串
const char* cutlass_status_to_string(my_cutlass_status_t status);

#ifdef __cplusplus
}  // extern "C"
#endif