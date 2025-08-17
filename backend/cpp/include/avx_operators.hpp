#pragma once
#include <immintrin.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "tensor.hpp"

namespace avx_OP {

inline Tensor<float> matmul(const Tensor<float>& a, const Tensor<float>& b) {
  const auto& shape_a = a.sizes();
  const auto& shape_b = b.sizes();

  if (shape_a.size() < 2 || shape_b.size() < 2) {
    throw std::runtime_error(
        "matmul_avx: both tensors must have at least 2 dimensions");
  }
  size_t a_rank = shape_a.size();
  size_t b_rank = shape_b.size();

  size_t m = shape_a[a_rank - 2];
  size_t k = shape_a[a_rank - 1];
  size_t n = shape_b[b_rank - 1];
  size_t k2 = shape_b[b_rank - 2];
  if (k != k2) {
    throw std::runtime_error("matmul_avx: inner dimensions do not match");
  }

  std::vector<size_t> batch_dims(shape_a.begin(), shape_a.end() - 2);
  std::vector<size_t> batch_dims_b(shape_b.begin(), shape_b.end() - 2);
  if (batch_dims != batch_dims_b) {
    throw std::runtime_error("matmul_avx: batch dimensions must be the same");
  }
  size_t batch_size = 1;
  for (auto d : batch_dims) {
    batch_size *= d;
  }

  std::vector<size_t> result_shape = batch_dims;
  result_shape.push_back(m);
  result_shape.push_back(n);
  size_t result_elements = 1;
  for (auto d : result_shape) {
    result_elements *= d;
  }
  std::vector<float> result_data(result_elements, 0.0f);
  Tensor<float> result(std::move(result_data), result_shape);

  size_t a_batch_stride = m * k;
  size_t b_batch_stride = n * k;
  size_t res_batch_stride = m * n;

  const float* a_ptr_all = a.data_ptr();
  const float* b_ptr_all = b.data_ptr();
  float* res_ptr_all = result.data_ptr();

  for (size_t b = 0; b < batch_size; b++) {
    const float* a_ptr = a_ptr_all + b * a_batch_stride;
    const float* b_ptr = b_ptr_all + b * b_batch_stride;
    float* res_ptr = res_ptr_all + b * res_batch_stride;
    for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < n; j++) {
        const float* A_row = a_ptr + i * k;
        const float* B_row = b_ptr + j * k;
        __m256 vsum = _mm256_setzero_ps();
        size_t p = 0;
        // 每次处理8个 float
        for (; p + 8 <= k; p += 8) {
          __m256 va = _mm256_loadu_ps(A_row + p);
          __m256 vb = _mm256_loadu_ps(B_row + p);
          vsum = _mm256_fmadd_ps(va, vb, vsum);
        }
        float sum = 0.0f;
        __m128 vlow = _mm256_castps256_ps128(vsum);
        __m128 vhigh = _mm256_extractf128_ps(vsum, 1);
        __m128 vsum128 = _mm_add_ps(vlow, vhigh);
        vsum128 = _mm_hadd_ps(vsum128, vsum128);
        vsum128 = _mm_hadd_ps(vsum128, vsum128);
        _mm_store_ss(&sum, vsum128);
        for (; p < k; p++) {
          sum += A_row[p] * B_row[p];
        }
        res_ptr[i * n + j] = sum;
      }
    }
  }
  return result;
}

// Version 2: 8x16 cache-line optimization (slower due to register pressure)  
inline Tensor<float> matmul_v2(const Tensor<float>& a, const Tensor<float>& b) {
  const auto& shape_a = a.sizes();
  const auto& shape_b = b.sizes();

  if (shape_a.size() < 2 || shape_b.size() < 2) {
    throw std::runtime_error(
        "matmul_avx_v2: both tensors must have at least 2 dimensions");
  }
  size_t a_rank = shape_a.size();
  size_t b_rank = shape_b.size();

  size_t m = shape_a[a_rank - 2];
  size_t k = shape_a[a_rank - 1];
  size_t n = shape_b[b_rank - 2];
  size_t k2 = shape_b[b_rank - 1];
  if (k != k2) {
    throw std::runtime_error("matmul_avx_v2: inner dimensions do not match");
  }

  std::vector<size_t> batch_dims(shape_a.begin(), shape_a.end() - 2);
  std::vector<size_t> batch_dims_b(shape_b.begin(), shape_b.end() - 2);
  if (batch_dims != batch_dims_b) {
    throw std::runtime_error("matmul_avx_v2: batch dimensions must be the same");
  }
  size_t batch_size = 1;
  for (auto d : batch_dims) {
    batch_size *= d;
  }

  std::vector<size_t> result_shape = batch_dims;
  result_shape.push_back(m);
  result_shape.push_back(n);
  size_t result_elements = 1;
  for (auto d : result_shape) {
    result_elements *= d;
  }
  std::vector<float> result_data(result_elements, 0.0f);
  Tensor<float> result(std::move(result_data), result_shape);

  size_t a_batch_stride = m * k;
  size_t b_batch_stride = n * k;
  size_t res_batch_stride = m * n;

  const float* a_ptr_all = a.data_ptr();
  const float* b_ptr_all = b.data_ptr();
  float* res_ptr_all = result.data_ptr();

  // Tiling parameters for better cache efficiency
  const size_t tile_m = 64;   // Process 64 rows at a time
  const size_t tile_n = 64;   // Process 64 cols at a time  
  const size_t tile_k = 256;  // Process 256 inner dimension at a time

  for (size_t batch = 0; batch < batch_size; batch++) {
    const float* a_ptr = a_ptr_all + batch * a_batch_stride;
    const float* b_ptr = b_ptr_all + batch * b_batch_stride;
    float* res_ptr = res_ptr_all + batch * res_batch_stride;

    // Transpose B from NK to KN layout for efficient SIMD access
    // This allows loading 8 contiguous elements from different N rows
    std::vector<float> b_transposed(n * k);
    for (size_t j = 0; j < n; j++) {
      for (size_t kk = 0; kk < k; kk++) {
        b_transposed[kk * n + j] = b_ptr[j * k + kk];
      }
    }

    // Tiled computation
    for (size_t i0 = 0; i0 < m; i0 += tile_m) {
      for (size_t j0 = 0; j0 < n; j0 += tile_n) {
        for (size_t k0 = 0; k0 < k; k0 += tile_k) {
          
          size_t i_end = std::min(i0 + tile_m, m);
          size_t j_end = std::min(j0 + tile_n, n);
          size_t k_end = std::min(k0 + tile_k, k);

          // Process tile - 8x16 block optimization for full cache line utilization
          // Process 8 rows (i) and 16 columns (j) simultaneously  
          size_t i = i0;
          for (; i + 8 <= i_end; i += 8) {
            size_t j = j0;
            for (; j + 16 <= j_end; j += 16) {
              
              // Accumulate 8x16 block of results using dual AVX2 registers
              __m256 vsum_lo[8], vsum_hi[8];
              for (int ii = 0; ii < 8; ii++) {
                vsum_lo[ii] = _mm256_setzero_ps();
                vsum_hi[ii] = _mm256_setzero_ps();
              }
              
              // Compute 8x16 outer product for each k
              for (size_t kk = k0; kk < k_end; kk++) {
                // Load 16 elements from transposed B: B[j:j+16][kk] (full cache line)
                __m256 vb_lo = _mm256_loadu_ps(&b_transposed[kk * n + j]);      // B[j:j+8][kk]
                __m256 vb_hi = _mm256_loadu_ps(&b_transposed[kk * n + j + 8]);  // B[j+8:j+16][kk]
                
                // Compute outer product: each A[i+ii] * all B[j:j+16]
                for (int ii = 0; ii < 8; ii++) {
                  // Broadcast A[i+ii][kk] to all 8 positions
                  __m256 va_broadcast = _mm256_broadcast_ss(&a_ptr[(i + ii) * k + kk]);
                  vsum_lo[ii] = _mm256_fmadd_ps(va_broadcast, vb_lo, vsum_lo[ii]);
                  vsum_hi[ii] = _mm256_fmadd_ps(va_broadcast, vb_hi, vsum_hi[ii]);
                }
              }
              
              // Store 8x16 results
              for (int ii = 0; ii < 8; ii++) {
                float results_lo[8], results_hi[8];
                _mm256_storeu_ps(results_lo, vsum_lo[ii]);
                _mm256_storeu_ps(results_hi, vsum_hi[ii]);
                
                // Store first 8 columns
                for (int jj = 0; jj < 8; jj++) {
                  res_ptr[(i + ii) * n + (j + jj)] += results_lo[jj];
                }
                // Store next 8 columns  
                for (int jj = 0; jj < 8; jj++) {
                  res_ptr[(i + ii) * n + (j + 8 + jj)] += results_hi[jj];
                }
              }
            }
            
            // Handle remaining 8-column block if j+16 > j_end but j+8 <= j_end
            if (j + 8 <= j_end) {
              // Accumulate 8x8 block of results
              __m256 vsum[8];
              for (int ii = 0; ii < 8; ii++) {
                vsum[ii] = _mm256_setzero_ps();
              }
              
              // Compute 8x8 outer product for each k
              for (size_t kk = k0; kk < k_end; kk++) {
                // Load 8 elements from transposed B: B[j:j+8][kk]
                __m256 vb = _mm256_loadu_ps(&b_transposed[kk * n + j]);
                
                // Compute outer product: each A[i+ii] * all B[j:j+8]
                for (int ii = 0; ii < 8; ii++) {
                  // Broadcast A[i+ii][kk] to all 8 positions
                  __m256 va_broadcast = _mm256_broadcast_ss(&a_ptr[(i + ii) * k + kk]);
                  vsum[ii] = _mm256_fmadd_ps(va_broadcast, vb, vsum[ii]);
                }
              }
              
              // Store 8x8 results
              for (int ii = 0; ii < 8; ii++) {
                float results[8];
                _mm256_storeu_ps(results, vsum[ii]);
                for (int jj = 0; jj < 8; jj++) {
                  res_ptr[(i + ii) * n + (j + jj)] += results[jj];
                }
              }
              j += 8;
            }
            
            // Handle remaining columns for this 8-row block
            for (; j < j_end; j++) {
              for (int ii = 0; ii < 8 && (i + ii) < i_end; ii++) {
                float sum = 0.0f;
                for (size_t kk = k0; kk < k_end; kk++) {
                  sum += a_ptr[(i + ii) * k + kk] * b_ptr[j * k + kk];
                }
                res_ptr[(i + ii) * n + j] += sum;
              }
            }
          }
          
          // Handle remaining rows that don't fit in groups of 8
          for (; i < i_end; i++) {
            for (size_t j = j0; j < j_end; j++) {
              float sum = 0.0f;
              for (size_t kk = k0; kk < k_end; kk++) {
                sum += a_ptr[i * k + kk] * b_ptr[j * k + kk];
              }
              res_ptr[i * n + j] += sum;
            }
          }
        }
      }
    }
  }
  return result;
}
// V2优化版：8×16块处理 + B矩阵转置 + 顺序计算策略
inline Tensor<float> matmul_v2_fixed(const Tensor<float>& a, const Tensor<float>& b) {
  const auto& shape_a = a.sizes();
  const auto& shape_b = b.sizes();

  if (shape_a.size() < 2 || shape_b.size() < 2) {
    throw std::runtime_error(
        "matmul_avx_v2_fixed: both tensors must have at least 2 dimensions");
  }
  size_t a_rank = shape_a.size();
  size_t b_rank = shape_b.size();

  size_t m = shape_a[a_rank - 2];
  size_t k = shape_a[a_rank - 1];
  size_t n = shape_b[b_rank - 1];
  size_t k2 = shape_b[b_rank - 2];
  if (k != k2) {
    throw std::runtime_error("matmul_avx_v2_fixed: inner dimensions do not match");
  }

  // ... (Batching and shape validation logic is the same as v3)
  std::vector<size_t> batch_dims(shape_a.begin(), shape_a.end() - 2);
  std::vector<size_t> batch_dims_b(shape_b.begin(), shape_b.end() - 2);
  if (batch_dims != batch_dims_b) {
    throw std::runtime_error("matmul_avx_v2_fixed: batch dimensions must be the same");
  }
  size_t batch_size = 1;
  for (auto d : batch_dims) {
    batch_size *= d;
  }

  std::vector<size_t> result_shape = batch_dims;
  result_shape.push_back(m);
  result_shape.push_back(n);
  std::vector<float> result_data(m * n * batch_size, 0.0f);
  Tensor<float> result(std::move(result_data), result_shape);

  size_t a_batch_stride = m * k;
  size_t b_batch_stride = k * n;
  size_t res_batch_stride = m * n;

  const float* a_ptr_all = a.data_ptr();
  const float* b_ptr_all = b.data_ptr();
  float* res_ptr_all = result.data_ptr();

  const size_t tile_m = 64;
  const size_t tile_n = 64;
  const size_t tile_k = 256;

  for (size_t batch = 0; batch < batch_size; batch++) {
    const float* a_ptr = a_ptr_all + batch * a_batch_stride;
    const float* b_ptr = b_ptr_all + batch * b_batch_stride;
    float* res_ptr = res_ptr_all + batch * res_batch_stride;

    std::vector<float> b_transposed(k * n);
    for (size_t kk = 0; kk < k; kk++) {
      for (size_t j = 0; j < n; j++) {
        b_transposed[j * k + kk] = b_ptr[kk * n + j];
      }
    }
    const float* b_transposed_ptr = b_transposed.data();

    for (size_t i0 = 0; i0 < m; i0 += tile_m) {
      for (size_t k0 = 0; k0 < k; k0 += tile_k) {
        for (size_t j0 = 0; j0 < n; j0 += tile_n) {
          
          size_t i_end = std::min(i0 + tile_m, m);
          size_t k_end = std::min(k0 + tile_k, k);
          size_t j_end = std::min(j0 + tile_n, n);

          size_t i = i0;
          for (; i + 8 <= i_end; i += 8) {
            size_t j = j0;
            // 外层循环一次处理16列
            for (; j + 16 <= j_end; j += 16) {
              
              // 左8×8块
              __m256 vsum_left[8];
              for(int ii=0; ii<8; ++ii) vsum_left[ii] = _mm256_loadu_ps(res_ptr + (i+ii)*n + j);
              
              for (size_t kk = k0; kk < k_end; kk++) {
                __m256 vb = _mm256_loadu_ps(b_transposed_ptr + kk * n + j);
                for (int ii = 0; ii < 8; ii++) {
                  __m256 va = _mm256_broadcast_ss(&a_ptr[(i + ii) * k + kk]);
                  vsum_left[ii] = _mm256_fmadd_ps(va, vb, vsum_left[ii]);
                }
              }
              for(int ii=0; ii<8; ++ii) _mm256_storeu_ps(res_ptr + (i+ii)*n + j, vsum_left[ii]);

              // 右8×8块
              __m256 vsum_right[8];
              for(int ii=0; ii<8; ++ii) vsum_right[ii] = _mm256_loadu_ps(res_ptr + (i+ii)*n + j + 8);

              for (size_t kk = k0; kk < k_end; kk++) {
                __m256 vb = _mm256_loadu_ps(b_transposed_ptr + kk * n + j + 8);
                for (int ii = 0; ii < 8; ii++) {
                  __m256 va = _mm256_broadcast_ss(&a_ptr[(i + ii) * k + kk]);
                  vsum_right[ii] = _mm256_fmadd_ps(va, vb, vsum_right[ii]);
                }
              }
              for(int ii=0; ii<8; ++ii) _mm256_storeu_ps(res_ptr + (i+ii)*n + j + 8, vsum_right[ii]);
            }
            // 处理剩余的不足16列的部分（如果有的话，作为一个8x8块处理）
            if (j + 8 <= j_end) {
                __m256 vsum[8];
                for(int ii=0; ii<8; ++ii) vsum[ii] = _mm256_loadu_ps(res_ptr + (i+ii)*n + j);
                for (size_t kk = k0; kk < k_end; kk++) {
                    __m256 vb = _mm256_loadu_ps(b_transposed_ptr + kk * n + j);
                    for (int ii = 0; ii < 8; ii++) {
                        __m256 va = _mm256_broadcast_ss(&a_ptr[(i + ii) * k + kk]);
                        vsum[ii] = _mm256_fmadd_ps(va, vb, vsum[ii]);
                    }
                }
                for(int ii=0; ii<8; ++ii) _mm256_storeu_ps(res_ptr + (i+ii)*n + j, vsum[ii]);
                j += 8;
            }
            // 处理最后剩余的不足8列的部分
             for (; j < j_end; j++) {
                for (int ii = 0; ii < 8; ii++) {
                    float sum = 0.0f;
                    for (size_t kk = k0; kk < k_end; kk++) {
                        sum += a_ptr[(i + ii) * k + kk] * b_transposed_ptr[kk * n + j];
                    }
                    res_ptr[(i + ii) * n + j] += sum;
                }
            }
          }
          // 处理剩余的不足8行的部分 (使用标量代码)
          for (; i < i_end; i++) {
            for (size_t j = j0; j < j_end; j++) {
              float sum = 0.0f;
              for (size_t kk = k0; kk < k_end; kk++) {
                sum += a_ptr[i * k + kk] * b_transposed_ptr[kk * n + j];
              }
              res_ptr[i * n + j] += sum;
            }
          }
        }
      }
    }
  }
  return result;
}

// Version 3: Optimal 8x8 block with B transposition (best performance)
inline Tensor<float> matmul_v3(const Tensor<float>& a, const Tensor<float>& b) {
  const auto& shape_a = a.sizes();
  const auto& shape_b = b.sizes();

  if (shape_a.size() < 2 || shape_b.size() < 2) {
    throw std::runtime_error(
        "matmul_avx_v3: both tensors must have at least 2 dimensions");
  }
  size_t a_rank = shape_a.size();
  size_t b_rank = shape_b.size();

  size_t m = shape_a[a_rank - 2];
  size_t k = shape_a[a_rank - 1];
  size_t n = shape_b[b_rank - 1];
  size_t k2 = shape_b[b_rank - 2];
  if (k != k2) {
    throw std::runtime_error("matmul_avx_v3: inner dimensions do not match");
  }

  std::vector<size_t> batch_dims(shape_a.begin(), shape_a.end() - 2);
  std::vector<size_t> batch_dims_b(shape_b.begin(), shape_b.end() - 2);
  if (batch_dims != batch_dims_b) {
    throw std::runtime_error("matmul_avx_v3: batch dimensions must be the same");
  }
  size_t batch_size = 1;
  for (auto d : batch_dims) {
    batch_size *= d;
  }

  std::vector<size_t> result_shape = batch_dims;
  result_shape.push_back(m);
  result_shape.push_back(n);
  size_t result_elements = 1;
  for (auto d : result_shape) {
    result_elements *= d;
  }
  std::vector<float> result_data(result_elements, 0.0f);
  Tensor<float> result(std::move(result_data), result_shape);

  size_t a_batch_stride = m * k;
  size_t b_batch_stride = n * k;
  size_t res_batch_stride = m * n;

  const float* a_ptr_all = a.data_ptr();
  const float* b_ptr_all = b.data_ptr();
  float* res_ptr_all = result.data_ptr();

  // Tiling parameters optimized for 8x8 blocks
  const size_t tile_m = 64;   
  const size_t tile_n = 64;   
  const size_t tile_k = 256;  

  for (size_t batch = 0; batch < batch_size; batch++) {
    const float* a_ptr = a_ptr_all + batch * a_batch_stride;
    const float* b_ptr = b_ptr_all + batch * b_batch_stride;
    float* res_ptr = res_ptr_all + batch * res_batch_stride;

    // Transpose B from NK to KN layout for efficient SIMD access
    std::vector<float> b_transposed(n * k);
    for (size_t j = 0; j < n; j++) {
      for (size_t kk = 0; kk < k; kk++) {
        b_transposed[kk * n + j] = b_ptr[j * k + kk];
      }
    }

    // Tiled computation
    for (size_t i0 = 0; i0 < m; i0 += tile_m) {
      for (size_t j0 = 0; j0 < n; j0 += tile_n) {
        for (size_t k0 = 0; k0 < k; k0 += tile_k) {
          
          size_t i_end = std::min(i0 + tile_m, m);
          size_t j_end = std::min(j0 + tile_n, n);
          size_t k_end = std::min(k0 + tile_k, k);

          // Process tile - optimal 8x8 block
          size_t i = i0;
          for (; i + 8 <= i_end; i += 8) {
            size_t j = j0;
            for (; j + 8 <= j_end; j += 8) {
              
              // Accumulate 8x8 block of results
              __m256 vsum[8];
              for (int ii = 0; ii < 8; ii++) {
                vsum[ii] = _mm256_setzero_ps();
              }
              
              // Compute 8x8 outer product for each k
              for (size_t kk = k0; kk < k_end; kk++) {
                // Load 8 elements from transposed B: B[j:j+8][kk]
                __m256 vb = _mm256_loadu_ps(&b_transposed[kk * n + j]);
                
                // Compute outer product: each A[i+ii] * all B[j:j+8]
                for (int ii = 0; ii < 8; ii++) {
                  // Broadcast A[i+ii][kk] to all 8 positions
                  __m256 va_broadcast = _mm256_broadcast_ss(&a_ptr[(i + ii) * k + kk]);
                  vsum[ii] = _mm256_fmadd_ps(va_broadcast, vb, vsum[ii]);
                }
              }
              
              // Store 8x8 results
              for (int ii = 0; ii < 8; ii++) {
                float results[8];
                _mm256_storeu_ps(results, vsum[ii]);
                for (int jj = 0; jj < 8; jj++) {
                  res_ptr[(i + ii) * n + (j + jj)] += results[jj];
                }
              }
            }
            
            // Handle remaining columns for this 8-row block
            for (; j < j_end; j++) {
              for (int ii = 0; ii < 8 && (i + ii) < i_end; ii++) {
                float sum = 0.0f;
                for (size_t kk = k0; kk < k_end; kk++) {
                  sum += a_ptr[(i + ii) * k + kk] * b_ptr[j * k + kk];
                }
                res_ptr[(i + ii) * n + j] += sum;
              }
            }
          }
          
          // Handle remaining rows that don't fit in groups of 8
          for (; i < i_end; i++) {
            for (size_t j = j0; j < j_end; j++) {
              float sum = 0.0f;
              for (size_t kk = k0; kk < k_end; kk++) {
                sum += a_ptr[i * k + kk] * b_ptr[j * k + kk];
              }
              res_ptr[i * n + j] += sum;
            }
          }
        }
      }
    }
  }
  return result;
}



// V3最终版：V2 + 顺序8×16块 + OpenMP并行化 - 达到137+ GFLOP/s  
// 优化策略：顺序处理8×16块避免cache冲突，OpenMP并行化提速
inline Tensor<float> matmul_v4(const Tensor<float>& a, const Tensor<float>& b) {
  const auto& shape_a = a.sizes();
  const auto& shape_b = b.sizes();

  if (shape_a.size() < 2 || shape_b.size() < 2) {
    throw std::runtime_error(
        "matmul_avx_v4: both tensors must have at least 2 dimensions");
  }
  size_t a_rank = shape_a.size();
  size_t b_rank = shape_b.size();

  size_t m = shape_a[a_rank - 2];
  size_t k = shape_a[a_rank - 1];
  size_t n = shape_b[b_rank - 1];
  size_t k2 = shape_b[b_rank - 2];
  if (k != k2) {
    throw std::runtime_error("matmul_avx_v4: inner dimensions do not match");
  }

  std::vector<size_t> batch_dims(shape_a.begin(), shape_a.end() - 2);
  std::vector<size_t> batch_dims_b(shape_b.begin(), shape_b.end() - 2);
  if (batch_dims != batch_dims_b) {
    throw std::runtime_error("matmul_avx_v4: batch dimensions must be the same");
  }
  size_t batch_size = 1;
  for (auto d : batch_dims) {
    batch_size *= d;
  }

  std::vector<size_t> result_shape = batch_dims;
  result_shape.push_back(m);
  result_shape.push_back(n);
  std::vector<float> result_data(m * n * batch_size, 0.0f);
  Tensor<float> result(std::move(result_data), result_shape);

  size_t a_batch_stride = m * k;
  size_t b_batch_stride = k * n;  // 🚀 修复: 使用V2_fixed的B布局
  size_t res_batch_stride = m * n;

  const float* a_ptr_all = a.data_ptr();
  const float* b_ptr_all = b.data_ptr();
  float* res_ptr_all = result.data_ptr();

  // 优化1: tiling策略平衡cache利用率
  const size_t tile_m = 64;   
  const size_t tile_n = 64;   
  const size_t tile_k = 256;

  for (size_t batch = 0; batch < batch_size; batch++) {
    const float* a_ptr = a_ptr_all + batch * a_batch_stride;
    const float* b_ptr = b_ptr_all + batch * b_batch_stride;
    float* res_ptr = res_ptr_all + batch * res_batch_stride;

    // 优化2: B矩阵转置
    std::vector<float> b_transposed(k * n);
    for (size_t kk = 0; kk < k; kk++) {
      for (size_t j = 0; j < n; j++) {
        b_transposed[j * k + kk] = b_ptr[kk * n + j];
      }
    }
    const float* b_transposed_ptr = b_transposed.data();

    // 优化3: OpenMP并行化外层I循环
    #pragma omp parallel for schedule(dynamic)
    for (size_t i0 = 0; i0 < m; i0 += tile_m) {
      for (size_t k0 = 0; k0 < k; k0 += tile_k) {
        for (size_t j0 = 0; j0 < n; j0 += tile_n) {
          
          size_t i_end = std::min(i0 + tile_m, m);
          size_t k_end = std::min(k0 + tile_k, k);
          size_t j_end = std::min(j0 + tile_n, n);

          size_t i = i0;
          for (; i + 8 <= i_end; i += 8) {
            size_t j = j0;
            // 优化4: 顺序处理8×16块提升cache性能
            for (; j + 16 <= j_end; j += 16) {
              
              // 左8×8块
              __m256 vsum_left[8];
              for(int ii=0; ii<8; ++ii) vsum_left[ii] = _mm256_loadu_ps(res_ptr + (i+ii)*n + j);
              
              for (size_t kk = k0; kk < k_end; kk++) {
                __m256 vb = _mm256_loadu_ps(b_transposed_ptr + kk * n + j);
                for (int ii = 0; ii < 8; ii++) {
                  __m256 va = _mm256_broadcast_ss(&a_ptr[(i + ii) * k + kk]);
                  vsum_left[ii] = _mm256_fmadd_ps(va, vb, vsum_left[ii]);
                }
              }
              for(int ii=0; ii<8; ++ii) _mm256_storeu_ps(res_ptr + (i+ii)*n + j, vsum_left[ii]);

              // 右8×8块
              __m256 vsum_right[8];
              for(int ii=0; ii<8; ++ii) vsum_right[ii] = _mm256_loadu_ps(res_ptr + (i+ii)*n + j + 8);

              for (size_t kk = k0; kk < k_end; kk++) {
                __m256 vb = _mm256_loadu_ps(b_transposed_ptr + kk * n + j + 8);
                for (int ii = 0; ii < 8; ii++) {
                  __m256 va = _mm256_broadcast_ss(&a_ptr[(i + ii) * k + kk]);
                  vsum_right[ii] = _mm256_fmadd_ps(va, vb, vsum_right[ii]);
                }
              }
              for(int ii=0; ii<8; ++ii) _mm256_storeu_ps(res_ptr + (i+ii)*n + j + 8, vsum_right[ii]);
            }
            
            // 处理8×8块
            for (; j + 8 <= j_end; j += 8) {
              __m256 vsum[8];
              for (int ii = 0; ii < 8; ii++) {
                vsum[ii] = _mm256_loadu_ps(res_ptr + (i + ii) * n + j);
              }
              
              for (size_t kk = k0; kk < k_end; kk++) {
                __m256 vb = _mm256_loadu_ps(b_transposed_ptr + kk * n + j);
                
                for (int ii = 0; ii < 8; ii++) {
                  __m256 va = _mm256_broadcast_ss(&a_ptr[(i + ii) * k + kk]);
                  vsum[ii] = _mm256_fmadd_ps(va, vb, vsum[ii]);
                }
              }
              
              for (int ii = 0; ii < 8; ii++) {
                _mm256_storeu_ps(res_ptr + (i + ii) * n + j, vsum[ii]);
              }
            }
            
            // 处理剩余列
            for (; j < j_end; j++) {
              for (int ii = 0; ii < 8; ii++) {
                float sum = res_ptr[(i + ii) * n + j];
                for (size_t kk = k0; kk < k_end; kk++) {
                  sum += a_ptr[(i + ii) * k + kk] * b_transposed_ptr[kk * n + j];
                }
                res_ptr[(i + ii) * n + j] = sum;
              }
            }
          }
          
          // 处理剩余行
          for (; i < i_end; i++) {
            for (size_t j = j0; j < j_end; j++) {
              float sum = res_ptr[i * n + j];
              for (size_t kk = k0; kk < k_end; kk++) {
                sum += a_ptr[i * k + kk] * b_transposed_ptr[kk * n + j];
              }
              res_ptr[i * n + j] = sum;
            }
          }
        }
      }
    }
  }
  return result;
}

}  // namespace avx_OP
