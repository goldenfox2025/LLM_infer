#pragma once

#include <cuda_runtime.h>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

// 前向声明
class CudaMemoryPool;
/**
 * CUDA内存池类
 *
 * 提供CUDA内存的高效管理，包括：
 * 1. 内存复用：避免频繁的cudaMalloc/cudaFree调用
 * 2. 内存缓存：保留已释放的内存块以供后续使用
 * 3. Prefill模式：为prefill阶段预分配一块连续内存
 */
class CudaMemoryPool {
 public:
  /**
   * 构造函数：初始化内存池并检查CUDA上下文
   */
  CudaMemoryPool() : is_shutting_down_(false), is_prefill_mode_(false),
                     is_prefill_phase_(false), prefill_buffer_(nullptr),
                     prefill_buffer_size_(0), prefill_buffer_used_(0),
                     prefill_max_size_(256 * 1024 * 1024) {
    // 初始化CUDA上下文
    cudaError_t err = cudaFree(0);
    // 忽略初始化错误
    if (err != cudaSuccess && err != cudaErrorInvalidDevicePointer) {
      // 初始化CUDA上下文失败
    }
  }

  /**
   * 析构函数：释放所有缓存的内存块
   *
   * 注意：对于全局单例，此函数通常不会被调用，因为单例会持续到程序结束
   */
  ~CudaMemoryPool() {
    std::lock_guard<std::mutex> lock(mutex_);
    is_shutting_down_ = true;

    // 检查CUDA驱动程序是否仍然可用
    cudaError_t driver_status = cudaFree(0);
    bool driver_available = (driver_status == cudaSuccess ||
                            driver_status == cudaErrorInvalidDevicePointer);

    if (driver_available) {
      // 释放常规内存池中的块
      for (auto& [size, blocks] : free_blocks_) {
        for (void* ptr : blocks) {
          cudaError_t err = cudaFree(ptr);
          if (err != cudaSuccess) {
            if (err == cudaErrorCudartUnloading) {
              driver_available = false;
              break;
            }
            // 释放缓存块失败
          }
        }
        if (!driver_available) break;
      }

      // 释放prefill模式的内存块
      if (driver_available && prefill_buffer_ != nullptr) {
        cudaError_t err = cudaFree(prefill_buffer_);
        if (err != cudaSuccess) {
          // 释放prefill缓冲区失败
        } else {
          prefill_buffer_ = nullptr;
          prefill_buffer_size_ = 0;
          prefill_buffer_used_ = 0;
        }
      }
    } else {
      // CUDA驱动程序已关闭，跳过内存释放
    }

    free_blocks_.clear();

    // 检查未释放的内存块
    if (!active_allocations_.empty()) {
      size_t active_bytes = 0;
      for (auto const& [ptr, size] : active_allocations_) {
        active_bytes += size;
      }
      // 发现未释放的内存块，可能存在内存泄漏
      active_allocations_.clear();
    }
  }

  // 禁止拷贝和移动操作
  CudaMemoryPool(const CudaMemoryPool&) = delete;
  CudaMemoryPool& operator=(const CudaMemoryPool&) = delete;
  CudaMemoryPool(CudaMemoryPool&&) = delete;
  CudaMemoryPool& operator=(CudaMemoryPool&&) = delete;

  /**
   * 分配内存
   *
   * @param size 请求的内存大小（字节）
   * @param is_prefill_request 是否是prefill阶段的请求
   * @return 分配的内存指针，失败时返回nullptr
   */
  void* allocate(size_t size, bool is_prefill_request = false) {
    if (size == 0) return nullptr;

    // 计算对齐后的大小（256字节对齐）
    size_t aligned_size = (size + 255) & ~255;
    std::lock_guard<std::mutex> lock(mutex_);

    // 检查是否处于关闭状态
    if (is_shutting_down_) {
      // 内存池正在关闭，无法分配内存
      return nullptr;
    }

    // 自动检测prefill请求
    if (!is_prefill_request && is_prefill_phase_) {
      is_prefill_request = true;
    }

    // 尝试从prefill buffer分配
    void* ptr = try_allocate_from_prefill(size, aligned_size, is_prefill_request);
    if (ptr) return ptr;

    // 尝试从缓存中分配
    ptr = try_allocate_from_cache(aligned_size);
    if (ptr) return ptr;


    // 直接从CUDA分配新内存
    return allocate_new_block(size, aligned_size);
  }

  /**
   * 尝试从prefill buffer分配内存
   */
  void* try_allocate_from_prefill(size_t size, size_t aligned_size, bool is_prefill_request) {
    // 如果不是prefill请求或prefill模式未启用，直接返回
    if (!is_prefill_request || !is_prefill_mode_ || !prefill_buffer_) {
      return nullptr;
    }

    // 检查是否有足够的空间
    if (prefill_buffer_used_ + aligned_size <= prefill_buffer_size_) {
      // 从prefill内存块中分配
      void* ptr = static_cast<char*>(prefill_buffer_) + prefill_buffer_used_;
      prefill_buffer_used_ += aligned_size;
      active_allocations_[ptr] = aligned_size;

      // 大型内存分配完成
      return ptr;
    }

    // 尝试扩展prefill buffer
    if (prefill_buffer_size_ < prefill_max_size_) {
      // 计算新大小
      size_t new_size = std::min(prefill_buffer_size_ * 3 / 2, prefill_max_size_);
      new_size = std::max(new_size, prefill_buffer_used_ + aligned_size);
      new_size = std::min(new_size, prefill_max_size_);

      // 分配新buffer
      void* new_buffer = nullptr;
      cudaError_t err = cudaMalloc(&new_buffer, new_size);
      if (err != cudaSuccess) {
        // 扩展prefill内存块失败
        return nullptr;
      }

      // 复制旧数据
      if (prefill_buffer_used_ > 0) {
        cudaError_t copy_err = cudaMemcpy(new_buffer, prefill_buffer_, prefill_buffer_used_, cudaMemcpyDeviceToDevice);
        if (copy_err != cudaSuccess) {
          // 复制prefill内存块数据失败
          cudaFree(new_buffer);
          return nullptr;
        }
      }

      // 释放旧buffer
      cudaFree(prefill_buffer_);

      // 更新prefill buffer信息
      prefill_buffer_ = new_buffer;
      prefill_buffer_size_ = new_size;

      // 从新buffer分配
      void* ptr = static_cast<char*>(prefill_buffer_) + prefill_buffer_used_;
      prefill_buffer_used_ += aligned_size;
      active_allocations_[ptr] = aligned_size;

      // prefill内存块已扩展
      return ptr;
    }

    return nullptr;
  }

  /**
   * 尝试从缓存中分配合适大小的块
   */
  void* try_allocate_from_cache(size_t aligned_size) {
    auto it = free_blocks_.find(aligned_size);
    if (it != free_blocks_.end() && !it->second.empty()) {
      void* ptr = it->second.back();
      it->second.pop_back();
      active_allocations_[ptr] = aligned_size;
      return ptr;
    }
    return nullptr;
  }



  /**
   * 直接从CUDA分配新内存块
   */
  void* allocate_new_block(size_t size, size_t aligned_size) {
    // 检查可用GPU内存
    size_t free_memory = 0, total_memory = 0;
    cudaError_t mem_err = cudaMemGetInfo(&free_memory, &total_memory);
    if (mem_err == cudaSuccess && aligned_size > free_memory * 0.8) {
      // 内存不足，尝试释放缓存
      trim_threshold_internal(2);
      cudaMemGetInfo(&free_memory, &total_memory);
      if (aligned_size > free_memory * 0.8) {
        trim_internal();
      }
    }

    // 分配新内存
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, aligned_size);
    if (err != cudaSuccess) {
      // cudaMalloc失败
      return nullptr;
    }

    // 记录分配
    active_allocations_[ptr] = aligned_size;
    return ptr;
  }

  /**
   * 释放内存回内存池
   *
   * @param ptr 要释放的内存指针
   */
  void free(void* ptr) {
    if (!ptr) return;

    std::lock_guard<std::mutex> lock(mutex_);
    // 关闭状态下直接释放
    if (is_shutting_down_) {
      free_during_shutdown(ptr);
      return;
    }
    // 检查指针是否由内存池管理
    auto it = active_allocations_.find(ptr);
    if (it == active_allocations_.end()) {
      // 尝试释放未管理的指针
      return;
    }
    size_t aligned_size = it->second;
    // 检查是否是prefill buffer中的内存
    if (is_from_prefill_buffer(ptr)) {
      free_from_prefill_buffer(ptr, aligned_size);
      active_allocations_.erase(it);
      return;
    }
    // 处理常规内存块
    if (should_cache_block(aligned_size)) {
      // 缓存内存块
      free_blocks_[aligned_size].push_back(ptr);
    } else {
      // 直接释放内存
      cudaError_t err = cudaFree(ptr);
      if (err != cudaSuccess) {
        handle_cuda_error(err, aligned_size);
      }
    }
    // 从活跃分配中移除
    active_allocations_.erase(it);
    // 定期清理过多的缓存
    perform_periodic_cleanup();
  }

  /**
   * 关闭状态下释放内存
   */
  void free_during_shutdown(void* ptr) {
    auto it = active_allocations_.find(ptr);
    if (it != active_allocations_.end()) {
      active_allocations_.erase(it);

      // 检查CUDA驱动程序是否可用
      cudaError_t driver_status = cudaFree(0);
      if (driver_status == cudaSuccess || driver_status == cudaErrorInvalidDevicePointer) {
        cudaFree(ptr); // 忽略错误
      }
    }
  }

  /**
   * 检查指针是否来自prefill buffer
   */
  bool is_from_prefill_buffer(void* ptr) {
    if (!is_prefill_mode_ || !prefill_buffer_) return false;

    char* prefill_start = static_cast<char*>(prefill_buffer_);
    char* prefill_end = prefill_start + prefill_buffer_size_;
    char* ptr_char = static_cast<char*>(ptr);

    return (ptr_char >= prefill_start && ptr_char < prefill_end);
  }

  /**
   * 释放prefill buffer中的内存
   */
  void free_from_prefill_buffer(void* ptr, size_t aligned_size) {
    // 不减少prefill_buffer_used_，因为prefill buffer是顺序分配的
    // 只有在reset_prefill_buffer()时才会重置prefill_buffer_used_
    // 释放prefill buffer中的内存块，但不减少已使用计数
  }

  /**
   * 判断是否应该缓存内存块
   */
  bool should_cache_block(size_t aligned_size) {
    // 根据块大小确定最大缓存数量
    size_t max_blocks = 8; // 默认为小块限制
    if (aligned_size >= 1024 * 1024) { // 大于1MB
      max_blocks = 2;
    } else if (aligned_size >= 65536) { // 大于64KB
      max_blocks = 4;
    }

    // 检查该大小类别的当前缓存数量
    return free_blocks_[aligned_size].size() < max_blocks;
  }

  /**
   * 处理CUDA错误
   */
  void handle_cuda_error(cudaError_t err, size_t aligned_size) {
    if (err == cudaErrorCudartUnloading) {
      is_shutting_down_ = true;
      // CUDA驱动程序正在关闭
    } else {
      // cudaFree失败
    }
  }

  /**
   * 执行定期清理
   */
  void perform_periodic_cleanup() {
    // 计算缓存的总块数
    size_t total_cached_blocks = 0;
    for (const auto& [size, blocks] : free_blocks_) {
      total_cached_blocks += blocks.size();
    }

    // 如果缓存的块总数超过阈值，执行部分清理
    if (total_cached_blocks > 100) {
      trim_threshold_internal(4); // 每个类别最多保留4个块
    }
  }

  // 手动释放缓存：释放指定大小类别或所有大小类别的所有空闲块
  // size = 0 表示释放所有缓存
  void trim(size_t size = 0) {
    std::lock_guard<std::mutex> lock(mutex_); // 加锁
    trim_internal(size);                      // 调用内部实现
  }

  // 手动释放缓存：对于每个大小类别，如果缓存的块数超过阈值，则释放多余的块
  void trim_threshold(size_t max_blocks_per_size) {
    std::lock_guard<std::mutex> lock(mutex_);       // 加锁
    trim_threshold_internal(max_blocks_per_size); // 调用内部实现
  }

  /**
   * Prefill模式相关方法
   */

  /**
   * 开启prefill模式，预分配一块内存用于prefill阶段
   *
   * @param initial_size 初始预分配的内存大小，默认为48MB
   * @param max_size prefill内存的最大大小，默认为128MB
   * @return 是否成功开启prefill模式
   */
  bool enable_prefill_mode(size_t initial_size = 48 * 1024 * 1024, size_t max_size = 128 * 1024 * 1024) {
    std::lock_guard<std::mutex> lock(mutex_);

    // 如果已经处于prefill模式，检查是否需要重新分配
    if (is_prefill_mode_) {
      if (prefill_buffer_ != nullptr && prefill_buffer_size_ >= initial_size) {
        // 已有buffer且大小合适，只需重置使用计数
        prefill_buffer_used_ = 0;
        prefill_max_size_ = max_size;
        // 已处于prefill模式，重置buffer
        return true;
      } else {
        // buffer不存在或大小不足，释放并重新分配
        // buffer不足，需要重新分配
        disable_prefill_mode();
      }
    }

    // 检查是否处于关闭状态
    if (is_shutting_down_) {
      // 内存池正在关闭，无法开启prefill模式
      return false;
    }

    // 检查可用GPU内存并调整大小
    initial_size = adjust_prefill_size(initial_size);

    // 分配prefill内存块
    cudaError_t err = cudaMalloc(&prefill_buffer_, initial_size);
    if (err != cudaSuccess) {
      std::cerr << "错误: 无法分配prefill内存块: "
                << (initial_size / (1024 * 1024)) << " MB，错误: "
                << cudaGetErrorString(err) << std::endl;
      prefill_buffer_ = nullptr;
      return false;
    }

    // 设置prefill模式参数
    prefill_buffer_size_ = initial_size;
    prefill_buffer_used_ = 0;
    prefill_max_size_ = max_size;
    is_prefill_mode_ = true;

    std::cerr << "已开启prefill模式: "
              << (prefill_buffer_size_ / (1024 * 1024)) << " MB，最大: "
              << (prefill_max_size_ / (1024 * 1024)) << " MB" << std::endl;

    return true;
  }

  /**
   * 调整prefill内存大小，确保不超过可用内存
   */
  size_t adjust_prefill_size(size_t requested_size) {
    size_t free_memory = 0, total_memory = 0;
    cudaError_t mem_err = cudaMemGetInfo(&free_memory, &total_memory);
    if (mem_err == cudaSuccess) {
      if (requested_size > free_memory * 0.8) {
        size_t adjusted_size = free_memory * 0.7;
        std::cerr << "警告: prefill内存大小过大，已调整为: "
                  << (adjusted_size / (1024 * 1024)) << " MB" << std::endl;
        return adjusted_size;
      }
    }
    return requested_size;
  }

  /**
   * 关闭prefill模式，释放预分配的内存
   */
  void disable_prefill_mode() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!is_prefill_mode_) return;

    // 释放prefill内存块
    if (prefill_buffer_ != nullptr) {
      // 检查CUDA驱动程序是否可用
      cudaError_t driver_status = cudaFree(0);
      if (driver_status == cudaSuccess || driver_status == cudaErrorInvalidDevicePointer) {
        cudaError_t err = cudaFree(prefill_buffer_);
        if (err != cudaSuccess) {
          std::cerr << "警告: 释放prefill内存块失败: " << cudaGetErrorString(err) << std::endl;
          if (err == cudaErrorCudartUnloading) {
            is_shutting_down_ = true;
          }
        }
      } else {
        is_shutting_down_ = true;
      }

      prefill_buffer_ = nullptr;
    }

    // 重置prefill模式参数
    prefill_buffer_size_ = 0;
    prefill_buffer_used_ = 0;
    is_prefill_mode_ = false;

    std::cerr << "已关闭prefill模式" << std::endl;
  }

  /**
   * 重置prefill buffer，但不释放内存
   * 这允许在多轮prefill操作之间重用已分配的内存
   */
  void reset_prefill_buffer() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!is_prefill_mode_ || prefill_buffer_ == nullptr) return;

    // 检查活跃分配中的prefill内存块
    auto [count, bytes] = count_active_prefill_allocations();

    // std::cerr << "重置prefill buffer前，使用情况: "
    //           << (prefill_buffer_used_ / (1024 * 1024)) << " MB 已使用, "
    //           << count << " 个活跃分配 ("
    //           << (bytes / (1024 * 1024)) << " MB)" << std::endl;

    prefill_buffer_used_ = 0;

    // std::cerr << "已重置prefill buffer，大小: "
    //           << (prefill_buffer_size_ / (1024 * 1024)) << " MB" << std::endl;
  }

  /**
   * 统计活跃的prefill内存分配
   *
   * @return pair<count, bytes> 活跃分配的数量和总字节数
   */
  std::pair<size_t, size_t> count_active_prefill_allocations() const {
    size_t count = 0;
    size_t bytes = 0;

    if (!prefill_buffer_) return {count, bytes};

    char* buffer_start = static_cast<char*>(prefill_buffer_);
    char* buffer_end = buffer_start + prefill_buffer_size_;

    for (const auto& [ptr, size] : active_allocations_) {
      char* ptr_char = static_cast<char*>(ptr);
      if (ptr_char >= buffer_start && ptr_char < buffer_end) {
        count++;
        bytes += size;
      }
    }

    return {count, bytes};
  }


  /**
   * 设置prefill阶段标志，用于自动检测是否应该使用prefill buffer
   *
   * @param is_prefill 是否处于prefill阶段
   */
  void set_prefill_phase(bool is_prefill) {
    std::lock_guard<std::mutex> lock(mutex_);
    is_prefill_phase_ = is_prefill;

    // 进入prefill阶段时，如果需要则启用prefill模式
    if (is_prefill && !is_prefill_mode_ && !is_shutting_down_) {
      enable_prefill_mode();
    }

    // 退出prefill阶段时，记录使用情况并重置buffer
    if (!is_prefill && is_prefill_mode_) {
      auto [count, bytes] = count_active_prefill_allocations();

      // std::cerr << "Prefill阶段结束，buffer使用情况: "
      //           << (prefill_buffer_used_ / (1024 * 1024)) << " MB / "
      //           << (prefill_buffer_size_ / (1024 * 1024)) << " MB" << std::endl;

      // std::cerr << "Prefill buffer中有 " << count
      //           << " 个活跃分配 (" << (bytes / (1024 * 1024))
      //           << " MB)，这些内存不会被立即释放" << std::endl;

      // 重置buffer，准备下一次使用
      prefill_buffer_used_ = 0;
    }
  }


  /**
   * 设置关闭标志，防止在程序退出过程中进行CUDA操作
   */
  void prepare_for_shutdown() {
    std::lock_guard<std::mutex> lock(mutex_);
    is_shutting_down_ = true;

    // 清空所有缓存的内存块，但不调用cudaFree
    free_blocks_.clear();

    // 不清空active_allocations_，因为这些内存块可能仍在使用
    // 操作系统会在程序退出时回收所有内存

    std::cerr << "内存池已准备好安全关闭" << std::endl;
  }
  /**
   * 检查是否处于关闭状态
   */
  bool is_shutting_down() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return is_shutting_down_;
  }
  /**
   * 获取内存池的统计信息结构体
   */
  struct PoolStats {
    size_t total_cached_blocks = 0; // 空闲列表中总块数
    size_t total_cached_bytes = 0;  // 空闲列表中总字节数 (GPU内存)
    size_t active_allocations = 0;  // 当前活跃（未释放回池）的块数
    size_t active_bytes = 0;        // 当前活跃的总字节数 (GPU内存)
    size_t size_categories = 0;     // 空闲列表中有多少种不同的大小类别
    size_t prefill_buffer_size = 0; // prefill内存块的总大小
    size_t prefill_buffer_used = 0; // 已使用的prefill内存大小
  };
  /**
   * 获取当前的统计信息
   */
  PoolStats getStats() const {
    std::lock_guard<std::mutex> lock(mutex_);

    PoolStats stats;
    stats.size_categories = free_blocks_.size();
    stats.active_allocations = active_allocations_.size();

    // 计算缓存的内存块
    for (const auto& [size, blocks] : free_blocks_) {
      stats.total_cached_blocks += blocks.size();
      stats.total_cached_bytes += size * blocks.size();
    }

    // 计算活跃分配的内存
    stats.active_bytes = std::accumulate(
        active_allocations_.begin(), active_allocations_.end(), 0ULL,
        [](size_t sum, const auto& pair) { return sum + pair.second; });

    // 添加prefill模式的统计信息
    stats.prefill_buffer_size = prefill_buffer_size_;
    stats.prefill_buffer_used = prefill_buffer_used_;

    return stats;
  }

 private:
  /**
   * 内部实现方法
   */

  /**
   * 释放所有或指定大小的缓存块
   *
   * @param size 要释放的块大小，0表示释放所有
   */
  void trim_internal(size_t size = 0) {
    if (size == 0) {
      // 释放所有缓存块
      for (auto& [block_size, blocks] : free_blocks_) {
        for (void* ptr : blocks) {
          cudaError_t err = cudaFree(ptr);
          if (err != cudaSuccess) {
            // 释放缓存块失败
          }
        }
      }
      free_blocks_.clear();
    } else {
      // 只释放指定大小的缓存块
      size_t aligned_size = (size + 255) & ~255;
      auto it = free_blocks_.find(aligned_size);
      if (it != free_blocks_.end()) {
        for (void* ptr : it->second) {
          cudaError_t err = cudaFree(ptr);
          if (err != cudaSuccess) {
            // 释放缓存块失败
          }
        }
        free_blocks_.erase(it);
      }
    }
  }

  /**
   * 限制每种大小的缓存块数量
   *
   * @param max_blocks_per_size 每种大小最多保留的块数
   */
  void trim_threshold_internal(size_t max_blocks_per_size) {
    for (auto it = free_blocks_.begin(); it != free_blocks_.end();) {
      size_t size = it->first;
      std::vector<void*>& blocks = it->second;

      // 如果块数量超过阈值，释放多余的块
      if (blocks.size() > max_blocks_per_size) {
        size_t blocks_to_release = blocks.size() - max_blocks_per_size;

        for (size_t i = 0; i < blocks_to_release; ++i) {
          if (blocks.empty()) break;

          void* ptr = blocks.back();
          cudaError_t err = cudaFree(ptr);
          if (err != cudaSuccess) {
            // 释放缓存块失败
          }
          blocks.pop_back();
        }
      }

      // 如果该大小类别变为空，从map中移除
      if (blocks.empty()) {
        it = free_blocks_.erase(it);
      } else {
        ++it;
      }
    }
  }

  /**
   * 成员变量
   */

  std::unordered_map<size_t, std::vector<void*>> free_blocks_;     // 空闲内存块映射
  std::unordered_map<void*, size_t> active_allocations_;           // 活跃内存块映射

  // Prefill模式相关
  bool is_prefill_mode_ = false;                                   // 是否处于prefill模式
  void* prefill_buffer_ = nullptr;                                 // prefill内存块指针
  size_t prefill_buffer_size_ = 0;                                 // prefill内存块大小
  size_t prefill_buffer_used_ = 0;                                 // prefill内存已使用大小
  size_t prefill_max_size_ = 256 * 1024 * 1024;                    // prefill内存最大大小(128MB)
  bool is_prefill_phase_ = false;                                  // 是否处于prefill阶段

  // 状态标志
  bool is_shutting_down_ = false;                                  // 是否处于关闭状态

  // 线程安全
  mutable std::mutex mutex_;                                       // 互斥锁
};


/**
 * 全局CUDA内存池单例
 *
 * 提供线程安全的全局访问点，确保内存池在整个程序生命周期内可用
 */
class GlobalCudaMemoryPool {
 public:
  /**
   * 获取全局唯一的CudaMemoryPool实例引用
   */
  static CudaMemoryPool& instance();

  /**
   * Prefill模式相关方法
   */

  /**
   * 开启prefill模式，预分配一块内存用于prefill阶段
   *
   * @param initial_size 初始预分配的内存大小，默认为48MB
   * @param max_size prefill内存的最大大小，默认为128MB
   * @return 是否成功开启prefill模式
   */
  static bool enable_prefill_mode(size_t initial_size = 48 * 1024 * 1024, size_t max_size = 128 * 1024 * 1024) {
    return instance().enable_prefill_mode(initial_size, max_size);
  }

  /**
   * 关闭prefill模式，释放预分配的内存
   */
  static void disable_prefill_mode() {
    instance().disable_prefill_mode();
  }

  /**
   * 重置prefill buffer，但不释放内存
   */
  static void reset_prefill_buffer() {
    instance().reset_prefill_buffer();
  }
  /**
   * 设置prefill阶段标志，用于自动检测是否应该使用prefill buffer
   */
  static void set_prefill_phase(bool is_prefill) {
    instance().set_prefill_phase(is_prefill);
  }
  /**
   * 设置关闭标志，防止在程序退出过程中进行CUDA操作
   */
  static void prepare_for_shutdown() {
    if (pool_instance_ptr != nullptr) {
      pool_instance_ptr->prepare_for_shutdown();
    }
  }

  /**
   * 检查是否处于关闭状态
   */
  static bool is_shutting_down() {
    if (pool_instance_ptr != nullptr) {
      return pool_instance_ptr->is_shutting_down();
    }
    return false;
  }

 private:
  // 禁止外部创建和复制
  GlobalCudaMemoryPool() = default;
  ~GlobalCudaMemoryPool() = default;
  GlobalCudaMemoryPool(const GlobalCudaMemoryPool&) = delete;
  GlobalCudaMemoryPool& operator=(const GlobalCudaMemoryPool&) = delete;
  GlobalCudaMemoryPool(GlobalCudaMemoryPool&&) = delete;
  GlobalCudaMemoryPool& operator=(GlobalCudaMemoryPool&&) = delete;

  // 全局单例实例
  static CudaMemoryPool* pool_instance_ptr;
  static std::once_flag init_flag_;
};