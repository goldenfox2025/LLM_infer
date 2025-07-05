#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <cuda_runtime.h>

// 纯C风格的CUTLASS接口，避免任何C++类型冲突
typedef enum {
    CUTLASS_C_SUCCESS = 0,
    CUTLASS_C_ERROR_INVALID_PROBLEM = 1,
    CUTLASS_C_ERROR_NOT_SUPPORTED = 2,
    CUTLASS_C_ERROR_WORKSPACE_NOT_AVAILABLE = 3,
    CUTLASS_C_ERROR_INTERNAL = 4
} cutlass_c_status_t;

typedef enum { CUTLASS_C_TYPE_BFLOAT16 = 0, CUTLASS_C_TYPE_HALF = 1, CUTLASS_C_TYPE_FLOAT = 2 } cutlass_c_data_type_t;

// 纯C风格的GEMM接口
cutlass_c_status_t cutlass_c_gemm_bfloat16(int m, int n, int k,
                                           const void* d_a,     // bfloat16 指针
                                           const void* d_b,     // bfloat16 指针
                                           const void* d_bias,  // bfloat16 指针，可为NULL
                                           void* d_d,           // bfloat16 指针
                                           cudaStream_t stream, float alpha);

// 获取CUTLASS状态的字符串描述
const char* cutlass_c_get_status_string(cutlass_c_status_t status);

#ifdef __cplusplus
}
#endif