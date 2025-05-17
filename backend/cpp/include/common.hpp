
#pragma once

// System includes
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>

// Standard library includes
#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

// Project includes
#include "tensor.hpp"

//------------------------------------------------------------------------------
// CUDA Error Checking Macros
//------------------------------------------------------------------------------

/**
 * @brief Check CUDA errors and throw an exception if an error occurred
 *
 * This macro checks the result of a CUDA API call and throws a runtime_error
 * if the call failed, including file and line information.
 */
#define CUDA_CHECK(call)                                                \
  do {                                                                  \
    cudaError_t err = call;                                             \
    if (err != cudaSuccess) {                                           \
      fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(err));                                 \
      throw std::runtime_error(cudaGetErrorString(err));                \
    }                                                                   \
  } while (0)

/**
 * @brief Alias for CUDA_CHECK for backward compatibility
 */
#define checkCudaErrors(call) CUDA_CHECK(call)

/**
 * @brief Check cuBLAS status and throw an exception if an error occurred
 */
#define CUBLAS_CHECK(call)                                                    \
  do {                                                                        \
    cublasStatus_t status = call;                                             \
    if (status != CUBLAS_STATUS_SUCCESS) {                                    \
      fprintf(stderr, "cuBLAS Error at %s:%d - %d\n", __FILE__, __LINE__,     \
              static_cast<int>(status));                                      \
      const char* error_string = "Unknown cuBLAS error";                      \
      switch (status) {                                                       \
        case CUBLAS_STATUS_NOT_INITIALIZED:                                   \
          error_string = "CUBLAS_STATUS_NOT_INITIALIZED";                     \
          break;                                                              \
        case CUBLAS_STATUS_ALLOC_FAILED:                                      \
          error_string = "CUBLAS_STATUS_ALLOC_FAILED";                        \
          break;                                                              \
        case CUBLAS_STATUS_INVALID_VALUE:                                     \
          error_string = "CUBLAS_STATUS_INVALID_VALUE";                       \
          break;                                                              \
        case CUBLAS_STATUS_ARCH_MISMATCH:                                     \
          error_string = "CUBLAS_STATUS_ARCH_MISMATCH";                       \
          break;                                                              \
        case CUBLAS_STATUS_MAPPING_ERROR:                                     \
          error_string = "CUBLAS_STATUS_MAPPING_ERROR";                       \
          break;                                                              \
        case CUBLAS_STATUS_EXECUTION_FAILED:                                  \
          error_string = "CUBLAS_STATUS_EXECUTION_FAILED";                    \
          break;                                                              \
        case CUBLAS_STATUS_INTERNAL_ERROR:                                    \
          error_string = "CUBLAS_STATUS_INTERNAL_ERROR";                      \
          break;                                                              \
        case CUBLAS_STATUS_NOT_SUPPORTED:                                     \
          error_string = "CUBLAS_STATUS_NOT_SUPPORTED";                       \
          break;                                                              \
        case CUBLAS_STATUS_LICENSE_ERROR:                                     \
          error_string = "CUBLAS_STATUS_LICENSE_ERROR";                       \
          break;                                                              \
      }                                                                       \
      throw std::runtime_error(std::string("cuBLAS error: ") + error_string); \
    }                                                                         \
  } while (0)

/**
 * @brief Helper function to check cuBLAS status
 *
 * This function can be used when you need to check cuBLAS status in a function
 * rather than using the macro directly.
 */
inline void checkCublasStatus(cublasStatus_t status, const char* file,
                              int line) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    char errorMsg[256];
    const char* error_string = "Unknown cuBLAS error";
    switch (status) {
      case CUBLAS_STATUS_NOT_INITIALIZED:
        error_string = "CUBLAS_STATUS_NOT_INITIALIZED";
        break;
      case CUBLAS_STATUS_ALLOC_FAILED:
        error_string = "CUBLAS_STATUS_ALLOC_FAILED";
        break;
      case CUBLAS_STATUS_INVALID_VALUE:
        error_string = "CUBLAS_STATUS_INVALID_VALUE";
        break;
      case CUBLAS_STATUS_ARCH_MISMATCH:
        error_string = "CUBLAS_STATUS_ARCH_MISMATCH";
        break;
      case CUBLAS_STATUS_MAPPING_ERROR:
        error_string = "CUBLAS_STATUS_MAPPING_ERROR";
        break;
      case CUBLAS_STATUS_EXECUTION_FAILED:
        error_string = "CUBLAS_STATUS_EXECUTION_FAILED";
        break;
      case CUBLAS_STATUS_INTERNAL_ERROR:
        error_string = "CUBLAS_STATUS_INTERNAL_ERROR";
        break;
      case CUBLAS_STATUS_NOT_SUPPORTED:
        error_string = "CUBLAS_STATUS_NOT_SUPPORTED";
        break;
      case CUBLAS_STATUS_LICENSE_ERROR:
        error_string = "CUBLAS_STATUS_LICENSE_ERROR";
        break;
    }
    snprintf(errorMsg, sizeof(errorMsg), "cuBLAS error %d (%s) at %s:%d",
             static_cast<int>(status), error_string, file, line);
    fprintf(stderr, "%s\n", errorMsg);
    throw std::runtime_error(errorMsg);
  }
}

/**
 * @brief Check CUTLASS status and throw an exception if an error occurred
 *
 * Note: This is commented out to avoid redefinition with cudaOP.cuh
 */
/*
#define CUTLASS_CHECK(status)                                             \
  {                                                                       \
    cutlass::Status error = status;                                       \
    if (error != cutlass::Status::kSuccess) {                             \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) \
                << " at: " << __FILE__ << ":" << __LINE__ << std::endl;   \
      throw std::runtime_error("CUTLASS error");                          \
    }                                                                     \
  }
*/

//------------------------------------------------------------------------------
// Debug Utilities
//------------------------------------------------------------------------------

/**
 * @brief Print tensor information for debugging
 *
 * @param tensor The tensor to print
 * @param tensor_name Name of the tensor for display
 * @param num_to_print Maximum number of elements to print
 */
template <typename T>
void debugPrintTensor(const Tensor<T>& tensor, const std::string& tensor_name,
                      size_t num_to_print = 10) {
  std::cout << "[Debug] " << tensor_name << ":\n";

  // 1) Print shape
  std::cout << "  shape: [";
  for (auto s : tensor.sizes()) {
    std::cout << s << " ";
  }
  std::cout << "]\n";

  // 2) Print strides
  std::cout << "  strides: [";
  for (auto st : tensor.strides()) {
    std::cout << st << " ";
  }
  std::cout << "]\n";

  // 3) Print device
  std::cout << "  device: ";
  if (tensor.device() == Device::CPU) {
    std::cout << "CPU";
  } else if (tensor.device() == Device::CUDA) {
    std::cout << "CUDA";
  } else {
    std::cout << "UNKNOWN";
  }
  std::cout << "\n";

  // 4) Print elements starting from offset 0
  size_t offset = 0;  // Start printing from the beginning
  size_t total_elements = tensor.numel();
  size_t n_print = std::min(num_to_print, total_elements - offset);

  std::cout << "  elements from offset " << offset << " (" << n_print
            << " element(s)): ";
  if (tensor.device() == Device::CPU) {
    const T* ptr = tensor.data_ptr();
    for (size_t i = 0; i < n_print; i++) {
      std::cout << ptr[offset + i] << " ";
    }
    std::cout << "\n";
  } else {
    // Copy from GPU to CPU, then print
    std::vector<T> host_buffer(n_print);
    cudaError_t err = cudaMemcpy(host_buffer.data(), tensor.data_ptr() + offset,
                                 n_print * sizeof(T), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      std::cout << "  [Error] cudaMemcpy failed\n";
      return;
    }
    for (size_t i = 0; i < n_print; i++) {
      std::cout << host_buffer[i] << " ";
    }
    std::cout << "\n";
  }
}

/**
 * @brief Specialized version of debugPrintTensor for __nv_bfloat16 type
 */
template <>
inline void debugPrintTensor<__nv_bfloat16>(const Tensor<__nv_bfloat16>& tensor,
                                            const std::string& tensor_name,
                                            size_t num_to_print) {
  std::cout << "[Debug] " << tensor_name << ":\n";

  // 1) Print shape
  std::cout << "  shape: [";
  for (auto s : tensor.sizes()) {
    std::cout << s << " ";
  }
  std::cout << "]\n";

  // 2) Print strides
  std::cout << "  strides: [";
  for (auto st : tensor.strides()) {
    std::cout << st << " ";
  }
  std::cout << "]\n";

  // 3) Print device
  std::cout << "  device: ";
  if (tensor.device() == Device::CPU) {
    std::cout << "CPU";
  } else if (tensor.device() == Device::CUDA) {
    std::cout << "CUDA";
  } else {
    std::cout << "UNKNOWN";
  }
  std::cout << "\n";

  // 4) Print elements starting from offset 0
  size_t offset = 0;
  size_t total_elements = tensor.numel();
  size_t n_print = std::min(num_to_print, total_elements - offset);

  std::cout << "  elements from offset " << offset << " (" << n_print
            << " element(s)): ";
  if (tensor.device() == Device::CPU) {
    const __nv_bfloat16* ptr = tensor.data_ptr();
    for (size_t i = 0; i < n_print; i++) {
      std::cout << static_cast<float>(ptr[offset + i]) << " ";
    }
    std::cout << "\n";
  } else {
    // Copy from GPU to CPU, then print
    std::vector<__nv_bfloat16> host_buffer(n_print);
    cudaError_t err =
        cudaMemcpy(host_buffer.data(), tensor.data_ptr() + offset,
                   n_print * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      std::cout << "  [Error] cudaMemcpy failed\n";
      return;
    }
    for (size_t i = 0; i < n_print; i++) {
      std::cout << static_cast<float>(host_buffer[i]) << " ";
    }
    std::cout << "\n";
  }
}

/**
 * @brief Specialized version of debugPrintTensor for __half type
 */
template <>
inline void debugPrintTensor<__half>(const Tensor<__half>& tensor,
                                     const std::string& tensor_name,
                                     size_t num_to_print) {
  std::cout << "[Debug] " << tensor_name << ":\n";

  // 1) Print shape
  std::cout << "  shape: [";
  for (auto s : tensor.sizes()) {
    std::cout << s << " ";
  }
  std::cout << "]\n";

  // 2) Print strides
  std::cout << "  strides: [";
  for (auto st : tensor.strides()) {
    std::cout << st << " ";
  }
  std::cout << "]\n";

  // 3) Print device
  std::cout << "  device: ";
  if (tensor.device() == Device::CPU) {
    std::cout << "CPU";
  } else if (tensor.device() == Device::CUDA) {
    std::cout << "CUDA";
  } else {
    std::cout << "UNKNOWN";
  }
  std::cout << "\n";

  // 4) Print elements starting from offset 0
  size_t offset = 0;
  size_t total_elements = tensor.numel();
  size_t n_print = std::min(num_to_print, total_elements - offset);

  std::cout << "  elements from offset " << offset << " (" << n_print
            << " element(s)): ";
  if (tensor.device() == Device::CPU) {
    const __half* ptr = tensor.data_ptr();
    for (size_t i = 0; i < n_print; i++) {
      std::cout << static_cast<float>(ptr[offset + i]) << " ";
    }
    std::cout << "\n";
  } else {
    // Copy from GPU to CPU, then print
    std::vector<__half> host_buffer(n_print);
    cudaError_t err =
        cudaMemcpy(host_buffer.data(), tensor.data_ptr() + offset,
                   n_print * sizeof(__half), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      std::cout << "  [Error] cudaMemcpy failed\n";
      return;
    }
    for (size_t i = 0; i < n_print; i++) {
      std::cout << static_cast<float>(host_buffer[i]) << " ";
    }
    std::cout << "\n";
  }
}

//------------------------------------------------------------------------------
// Timing Utilities
//------------------------------------------------------------------------------

/**
 * @brief GPU timer for measuring kernel execution time
 *
 * Uses CUDA events to measure elapsed time on the GPU
 */
class GpuTimer {
 private:
  cudaEvent_t start_;
  cudaEvent_t stop_;

 public:
  GpuTimer() {
    CUDA_CHECK(cudaEventCreate(&start_));
    CUDA_CHECK(cudaEventCreate(&stop_));
  }

  ~GpuTimer() noexcept {
    // 在析构函数中不抛出异常，而是打印错误信息
    cudaError_t err = cudaEventDestroy(start_);
    if (err != cudaSuccess) {
      fprintf(stderr, "CUDA Error in ~GpuTimer(): %s\n",
              cudaGetErrorString(err));
    }

    err = cudaEventDestroy(stop_);
    if (err != cudaSuccess) {
      fprintf(stderr, "CUDA Error in ~GpuTimer(): %s\n",
              cudaGetErrorString(err));
    }
  }

  void start(cudaStream_t stream = nullptr) {
    CUDA_CHECK(cudaEventRecord(start_, stream));
  }

  void stop(cudaStream_t stream = nullptr) {
    CUDA_CHECK(cudaEventRecord(stop_, stream));
  }

  float milliseconds() {
    CUDA_CHECK(cudaEventSynchronize(stop_));
    float time;
    CUDA_CHECK(cudaEventElapsedTime(&time, start_, stop_));
    return time;
  }

  float seconds() { return milliseconds() * 1e-3f; }
};

/**
 * @brief CPU timer for measuring execution time
 *
 * Uses std::chrono to measure elapsed time on the CPU
 */
class CpuTimer {
 private:
  std::chrono::high_resolution_clock::time_point start_;
  std::chrono::high_resolution_clock::time_point stop_;

 public:
  void start() { start_ = std::chrono::high_resolution_clock::now(); }

  void stop() { stop_ = std::chrono::high_resolution_clock::now(); }

  double milliseconds() {
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(stop_ - start_);
    return duration.count() / 1000.0;
  }

  double seconds() { return milliseconds() / 1000.0; }
};

//------------------------------------------------------------------------------
// Thread Management Utilities
//------------------------------------------------------------------------------

/**
 * @brief Bind the current thread to a specific CPU core
 *
 * @param core_id The ID of the core to bind to
 */
void bind_this_thread_to_core(int core_id) {
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(core_id, &cpuset);
  pthread_t current_thread = pthread_self();

  int rc = pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
  if (rc != 0) {
    std::cerr << "Error calling pthread_setaffinity_np: " << strerror(rc)
              << "\n";
  }
}

/**
 * @brief Get the number of available CPU cores
 *
 * @return The number of CPU cores
 */
int get_num_cores() { return sysconf(_SC_NPROCESSORS_ONLN); }

//------------------------------------------------------------------------------
// Memory Management Utilities
//------------------------------------------------------------------------------

/**
 * @brief Print current CUDA memory usage
 *
 * @param location A string to identify where this function is called from
 */
void print_cuda_memory_usage(const char* location = "Current") {
  size_t free_memory = 0, total_memory = 0;
  CUDA_CHECK(cudaMemGetInfo(&free_memory, &total_memory));

  float free_gb = free_memory / (1024.0f * 1024.0f * 1024.0f);
  float total_gb = total_memory / (1024.0f * 1024.0f * 1024.0f);
  float used_gb = total_gb - free_gb;

  printf("[%s] CUDA Memory: Used = %.2f GB, Free = %.2f GB, Total = %.2f GB\n",
         location, used_gb, free_gb, total_gb);
}

/**
 * @brief Allocate device memory
 *
 * @param size Size in bytes to allocate
 * @return Pointer to allocated memory
 */
inline void* cuda_malloc(size_t size) {
  void* ptr = nullptr;
  CUDA_CHECK(cudaMalloc(&ptr, size));
  return ptr;
}

/**
 * @brief Free device memory
 *
 * @param ptr Pointer to memory to free
 */
inline void cuda_free(void* ptr) {
  if (ptr) {
    CUDA_CHECK(cudaFree(ptr));
  }
}

/**
 * @brief Copy memory from host to device
 *
 * @param dst Destination pointer (device)
 * @param src Source pointer (host)
 * @param size Size in bytes to copy
 * @param stream CUDA stream to use (optional)
 */
inline void cuda_h2d(void* dst, const void* src, size_t size,
                     cudaStream_t stream = nullptr) {
  if (stream) {
    CUDA_CHECK(cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream));
  } else {
    CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
  }
}

/**
 * @brief Copy memory from device to host
 *
 * @param dst Destination pointer (host)
 * @param src Source pointer (device)
 * @param size Size in bytes to copy
 * @param stream CUDA stream to use (optional)
 */
inline void cuda_d2h(void* dst, const void* src, size_t size,
                     cudaStream_t stream = nullptr) {
  if (stream) {
    CUDA_CHECK(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, stream));
  } else {
    CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
  }
}

/**
 * @brief Copy memory from device to device
 *
 * @param dst Destination pointer (device)
 * @param src Source pointer (device)
 * @param size Size in bytes to copy
 * @param stream CUDA stream to use (optional)
 */
inline void cuda_d2d(void* dst, const void* src, size_t size,
                     cudaStream_t stream = nullptr) {
  if (stream) {
    CUDA_CHECK(
        cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, stream));
  } else {
    CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice));
  }
}

/**
 * @brief Set device memory to a value
 *
 * @param ptr Pointer to memory to set
 * @param value Value to set
 * @param size Size in bytes to set
 * @param stream CUDA stream to use (optional)
 */
inline void cuda_memset(void* ptr, int value, size_t size,
                        cudaStream_t stream = nullptr) {
  if (stream) {
    CUDA_CHECK(cudaMemsetAsync(ptr, value, size, stream));
  } else {
    CUDA_CHECK(cudaMemset(ptr, value, size));
  }
}

/**
 * @brief Simple memory unit for managing host and device memory
 *
 * Provides utilities for allocating, copying, and freeing memory
 */
template <typename T>
struct MemoryUnit {
  T* host_ptr;
  T* device_ptr;
  size_t size_bytes;
  size_t elements;

  MemoryUnit(size_t elements_)
      : size_bytes(elements_ * sizeof(T)), elements(elements_) {
    host_ptr = static_cast<T*>(malloc(elements_ * sizeof(T)));
    if (!host_ptr) {
      throw std::runtime_error("Failed to allocate host memory");
    }
    device_ptr = static_cast<T*>(cuda_malloc(elements_ * sizeof(T)));
  }

  ~MemoryUnit() { free_all(); }

  void h2d(cudaStream_t stream = nullptr) {
    cuda_h2d(device_ptr, host_ptr, size_bytes, stream);
  }

  void d2h(cudaStream_t stream = nullptr) {
    cuda_d2h(host_ptr, device_ptr, size_bytes, stream);
  }

  void free_all() {
    if (host_ptr) {
      free(host_ptr);
      host_ptr = nullptr;
    }
    if (device_ptr) {
      cuda_free(device_ptr);
      device_ptr = nullptr;
    }
  }

  void init(int abs_range = 1) {
    for (size_t i = 0; i < elements; i++) {
      host_ptr[i] = static_cast<T>(
          rand() % 100 / static_cast<float>(100) * 2 * abs_range - abs_range);
    }
    h2d();
  }

  // 重新分配内存，保留原有数据
  void resize(size_t new_elements) {
    if (new_elements == elements) {
      return;
    }

    // 分配新的主机内存
    T* new_host_ptr = static_cast<T*>(malloc(new_elements * sizeof(T)));
    if (!new_host_ptr) {
      throw std::runtime_error("Failed to allocate host memory during resize");
    }

    // 复制原有数据
    size_t copy_elements = std::min(elements, new_elements);
    memcpy(new_host_ptr, host_ptr, copy_elements * sizeof(T));

    // 分配新的设备内存
    T* new_device_ptr = static_cast<T*>(cuda_malloc(new_elements * sizeof(T)));

    // 复制原有数据到设备
    if (copy_elements > 0) {
      cuda_d2d(new_device_ptr, device_ptr, copy_elements * sizeof(T));
    }

    // 释放旧内存
    free(host_ptr);
    cuda_free(device_ptr);

    // 更新指针和大小
    host_ptr = new_host_ptr;
    device_ptr = new_device_ptr;
    elements = new_elements;
    size_bytes = new_elements * sizeof(T);
  }

  // 清零设备内存
  void zero_device(cudaStream_t stream = nullptr) {
    cuda_memset(device_ptr, 0, size_bytes, stream);
  }

  // 清零主机内存
  void zero_host() { memset(host_ptr, 0, size_bytes); }

  // 清零所有内存
  void zero_all(cudaStream_t stream = nullptr) {
    zero_host();
    zero_device(stream);
  }
};

//------------------------------------------------------------------------------
// Miscellaneous Utilities
//------------------------------------------------------------------------------

/**
 * @brief Check if a number is a power of 2
 *
 * @param x The number to check
 * @return true if x is a power of 2, false otherwise
 */
template <typename T>
bool is_power_of_2(T x) {
  return x > 0 && (x & (x - 1)) == 0;
}

/**
 * @brief Calculate the greatest common divisor of two numbers
 *
 * @param a First number
 * @param b Second number
 * @return The greatest common divisor
 */
template <typename T>
T gcd(T a, T b) {
  while (b != 0) {
    T temp = b;
    b = a % b;
    a = temp;
  }
  return a;
}

/**
 * @brief Calculate the least common multiple of two numbers
 *
 * @param a First number
 * @param b Second number
 * @return The least common multiple
 */
template <typename T>
T lcm(T a, T b) {
  return (a / gcd(a, b)) * b;
}

/**
 * @brief Round up to the next multiple of a given number
 *
 * @param value The value to round up
 * @param multiple The multiple to round up to
 * @return The rounded up value
 */
template <typename T>
T round_up(T value, T multiple) {
  if (multiple == 0) return value;
  T remainder = value % multiple;
  if (remainder == 0) return value;
  return value + multiple - remainder;
}

//------------------------------------------------------------------------------
// CUDA-specific Utilities
//------------------------------------------------------------------------------

/**
 * @brief Get CUDA device properties
 *
 * @param device_id The device ID
 * @return The device properties
 */
cudaDeviceProp get_device_properties(int device_id = 0) {
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
  return prop;
}

/**
 * @brief Print CUDA device information
 *
 * @param device_id The device ID
 */
void print_device_info(int device_id = 0) {
  cudaDeviceProp prop = get_device_properties(device_id);

  printf("Device: %s\n", prop.name);
  printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
  printf(
      "  Total Global Memory: %.2f GB\n",
      static_cast<float>(prop.totalGlobalMem) / (1024.0f * 1024.0f * 1024.0f));
  printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
  printf("  Max Threads Dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0],
         prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
  printf("  Max Grid Size: (%d, %d, %d)\n", prop.maxGridSize[0],
         prop.maxGridSize[1], prop.maxGridSize[2]);
  printf("  Warp Size: %d\n", prop.warpSize);
  printf("  Memory Clock Rate: %.0f MHz\n", prop.memoryClockRate / 1000.0f);
  printf("  Memory Bus Width: %d bits\n", prop.memoryBusWidth);
  printf("  L2 Cache Size: %d KB\n", prop.l2CacheSize / 1024);
}

/**
 * @brief Initialize CUDA device
 *
 * @param device_id The device ID to initialize
 * @param print_info Whether to print device information
 */
void init_cuda_device(int device_id = 0, bool print_info = false) {
  CUDA_CHECK(cudaSetDevice(device_id));

  // Optional: Set cache configuration
  CUDA_CHECK(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

  if (print_info) {
    print_device_info(device_id);
  }
}

/**
 * @brief Get the number of CUDA devices
 *
 * @return The number of CUDA devices
 */
int get_cuda_device_count() {
  int count;
  CUDA_CHECK(cudaGetDeviceCount(&count));
  return count;
}

/**
 * @brief Get the current CUDA device
 *
 * @return The current device ID
 */
int get_current_cuda_device() {
  int device;
  CUDA_CHECK(cudaGetDevice(&device));
  return device;
}

/**
 * @brief Synchronize the current CUDA device
 */
void sync_cuda_device() { CUDA_CHECK(cudaDeviceSynchronize()); }

/**
 * @brief Reset the current CUDA device
 */
void reset_cuda_device() { CUDA_CHECK(cudaDeviceReset()); }

//------------------------------------------------------------------------------
// Debug and Logging Utilities
//------------------------------------------------------------------------------

/**
 * @brief Enum for log levels
 */
enum class LogLevel { DEBUG, INFO, WARNING, ERROR, FATAL };

/**
 * @brief Simple logger class
 */
class Logger {
 private:
  static LogLevel current_level_;
  static std::mutex log_mutex_;

  static const char* level_to_string(LogLevel level) {
    switch (level) {
      case LogLevel::DEBUG:
        return "DEBUG";
      case LogLevel::INFO:
        return "INFO";
      case LogLevel::WARNING:
        return "WARNING";
      case LogLevel::ERROR:
        return "ERROR";
      case LogLevel::FATAL:
        return "FATAL";
      default:
        return "UNKNOWN";
    }
  }

 public:
  static void set_level(LogLevel level) { current_level_ = level; }

  static LogLevel get_level() { return current_level_; }

  template <typename... Args>
  static void log(LogLevel level, const char* format, Args... args) {
    if (level < current_level_) return;

    std::lock_guard<std::mutex> lock(log_mutex_);

    // Get current time
    auto now = std::chrono::system_clock::now();
    auto now_c = std::chrono::system_clock::to_time_t(now);
    char time_buf[20];
    std::strftime(time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M:%S",
                  std::localtime(&now_c));

    // Print header
    fprintf(stderr, "[%s] [%s] ", time_buf, level_to_string(level));

    // Print message
    fprintf(stderr, format, args...);
    fprintf(stderr, "\n");

    // If fatal, exit
    if (level == LogLevel::FATAL) {
      exit(EXIT_FAILURE);
    }
  }

  template <typename... Args>
  static void debug(const char* format, Args... args) {
    log(LogLevel::DEBUG, format, args...);
  }

  template <typename... Args>
  static void info(const char* format, Args... args) {
    log(LogLevel::INFO, format, args...);
  }

  template <typename... Args>
  static void warning(const char* format, Args... args) {
    log(LogLevel::WARNING, format, args...);
  }

  template <typename... Args>
  static void error(const char* format, Args... args) {
    log(LogLevel::ERROR, format, args...);
  }

  template <typename... Args>
  static void fatal(const char* format, Args... args) {
    log(LogLevel::FATAL, format, args...);
  }
};

// Initialize static members
LogLevel Logger::current_level_ = LogLevel::INFO;
std::mutex Logger::log_mutex_;

//------------------------------------------------------------------------------
// Printing and Formatting Utilities
//------------------------------------------------------------------------------

/**
 * @brief Print a vector
 *
 * @param vec The vector to print
 * @param name The name of the vector
 */
template <typename T>
void print_vector(const std::vector<T>& vec, const std::string& name = "") {
  if (!name.empty()) {
    std::cout << name << ": ";
  }

  std::cout << "[";
  for (size_t i = 0; i < vec.size(); ++i) {
    std::cout << vec[i];
    if (i < vec.size() - 1) {
      std::cout << ", ";
    }
  }
  std::cout << "]" << std::endl;
}

/**
 * @brief Print a shape (vector of dimensions)
 *
 * @param shape The shape to print
 * @param name The name of the shape
 */
template <typename T>
void print_shape(const std::vector<T>& shape, const std::string& name = "") {
  if (!name.empty()) {
    std::cout << name << " shape: ";
  }

  std::cout << "[";
  if (shape.empty()) {
    std::cout << "<empty>";
  } else {
    for (size_t i = 0; i < shape.size(); ++i) {
      std::cout << shape[i];
      if (i < shape.size() - 1) {
        std::cout << ", ";
      }
    }
  }
  std::cout << "]" << std::endl;
}

/**
 * @brief Format a number with commas as thousands separators
 *
 * @param value The number to format
 * @return The formatted string
 */
template <typename T>
std::string format_with_commas(T value) {
  std::string result;
  std::string str = std::to_string(value);

  int count = 0;
  for (auto it = str.rbegin(); it != str.rend(); ++it) {
    if (count > 0 && count % 3 == 0) {
      result.push_back(',');
    }
    result.push_back(*it);
    ++count;
  }

  std::reverse(result.begin(), result.end());
  return result;
}

/**
 * @brief Format a size in bytes to a human-readable string
 *
 * @param bytes The size in bytes
 * @return The formatted string
 */
std::string format_bytes(size_t bytes) {
  static const char* suffixes[] = {"B", "KB", "MB", "GB", "TB", "PB"};

  int suffix_idx = 0;
  double size = static_cast<double>(bytes);

  while (size >= 1024 && suffix_idx < 5) {
    size /= 1024;
    ++suffix_idx;
  }

  char buffer[32];
  if (size - static_cast<size_t>(size) == 0) {
    snprintf(buffer, sizeof(buffer), "%zu %s", static_cast<size_t>(size),
             suffixes[suffix_idx]);
  } else {
    snprintf(buffer, sizeof(buffer), "%.2f %s", size, suffixes[suffix_idx]);
  }

  return std::string(buffer);
}