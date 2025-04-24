#pragma once // 防止头文件被多次包含

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>    // 用于 std::atexit (虽然最终没用，但保留包含可能有用)
#include <execinfo.h> // 用于获取调用栈 (如果需要更详细的调试)
#include <iostream>
#include <mutex>      // 用于 std::mutex 和 std::lock_guard / std::call_once
#include <numeric>    // 需要包含 numeric 用于 std::accumulate
#include <string>
#include <system_error> // 用于 std::call_once
#include <unordered_map> // 用于存储内存块
#include <vector>
#include <mutex> // 用于 std::once_flag (较新写法，但可能需要 C++17 或更高，回退到 <mutex>)

// 定义 CUDA 内存池类
class CudaMemoryPool {
 public:
  // 构造函数：尝试初始化 CUDA 上下文
  CudaMemoryPool() {
    // 尝试一个无操作的 CUDA 调用（如 cudaFree(0) 或 cudaGetDevice）来确保 CUDA
    // 上下文被初始化。 这在多线程或延迟初始化环境中尤其重要。
    cudaError_t err = cudaFree(0); // cudaFree(nullptr) 是合法的无操作调用
    if (err != cudaSuccess) {
      // 如果 cudaFree(0) 返回错误（除了正常情况），打印警告。
      // 注意: 在没有设备或驱动的情况下，这里可能会失败。
      // 在某些系统上，cudaFree(0) 可能返回 cudaErrorInvalidDevicePointer，这通常是无害的。
       if (err != cudaErrorInvalidDevicePointer) {
            std::cerr << "警告: CudaMemoryPool 构造函数中的 CUDA 上下文初始化检查返回: "
                      << cudaGetErrorString(err) << std::endl;
       }
    }
    // 或者使用 cudaGetDevice(&device_id); 检查是否能获取设备 ID
    // int current_device;
    // cudaError_t err_get_device = cudaGetDevice(¤t_device);
    // if (err_get_device != cudaSuccess) {
    //     std::cerr << "警告: CudaMemoryPool 无法获取当前 CUDA 设备: "
    //               << cudaGetErrorString(err_get_device) << std::endl;
    // }
  }

  // 析构函数：释放所有缓存的内存块
  // 注意：对于通过 GlobalCudaMemoryPool::instance() 获取的全局单例，
  // 由于采用了特定的生命周期管理策略（不显式删除），这个析构函数在正常程序退出时不会被调用。
  // 但是，如果用户直接创建 CudaMemoryPool 对象，则此析构函数会在对象销毁时执行。
  ~CudaMemoryPool() {
    //std::cerr << "CudaMemoryPool 析构函数被调用（可能不是全局单例）" << std::endl;
    std::lock_guard<std::mutex> lock(mutex_);

    // 释放所有大小类别中的缓存（空闲）内存块
    size_t freed_cached_bytes = 0;
    size_t freed_cached_count = 0;
    for (auto& [size, blocks] : free_blocks_) {
      freed_cached_count += blocks.size();
      freed_cached_bytes += size * blocks.size();
      for (void* ptr : blocks) {
        cudaError_t err = cudaFree(ptr);
        if (err != cudaSuccess) {
             std::cerr << "警告: CudaMemoryPool 析构函数 cudaFree 缓存块失败 (大小: " << size << "): "
                       << cudaGetErrorString(err) << std::endl;
        }
      }
    }
    //std::cerr << "CudaMemoryPool 析构函数: 释放了 " << freed_cached_count << " 个缓存块 (" << freed_cached_bytes << " 字节)" << std::endl;
    free_blocks_.clear(); // 清空空闲块记录

    // 对仍然活跃的（未被 free 回池的）内存块发出警告
    // 在正常情况下，如果所有分配的内存都已正确 free，这里应该是空的。
    // 如果这个 CudaMemoryPool 实例被销毁时还有活跃分配，说明存在内存泄漏（未调用 free）。
    if (!active_allocations_.empty()) {
      size_t active_bytes = 0;
       for (auto const& [ptr, size] : active_allocations_) {
           active_bytes += size;
       }
      std::cerr << "警告: CudaMemoryPool 析构函数发现 "
                << active_allocations_.size() << " 个活跃分配 ("
                << active_bytes << " 字节) 未被释放回池。可能存在内存泄漏！"
                << std::endl;
      // 在析构函数中不尝试释放活跃分配，因为它们可能仍然被程序的其他部分（错误地）引用。
      // 让程序的终止或用户的调试来处理这些泄漏。
      // for (auto const& [ptr, size] : active_allocations_) {
      //   cudaFree(ptr); // 通常不建议在这里强制释放，可能隐藏真正的问题
      // }
      active_allocations_.clear(); // 清空活跃分配记录
    }
  }

  // 禁止拷贝构造和拷贝赋值
  CudaMemoryPool(const CudaMemoryPool&) = delete;
  CudaMemoryPool& operator=(const CudaMemoryPool&) = delete;
  // 禁止移动构造和移动赋值（内存池通常不适合移动）
  CudaMemoryPool(CudaMemoryPool&&) = delete;
  CudaMemoryPool& operator=(CudaMemoryPool&&) = delete;

  // 分配内存方法
  void* allocate(size_t size) {
    if (size == 0) {
      // 分配 0 字节是无意义的，返回 nullptr
      return nullptr;
    }

    // 计算对齐后的大小：将请求的大小向上对齐到 256 字节的倍数。
    // CUDA 分配通常有固有的对齐要求，256 是一个常用且相对安全的对齐值。
    // (size + 255) & ~255 是一种高效的向上对齐到位掩码操作。
    size_t aligned_size = (size + 255) & ~255;

    std::lock_guard<std::mutex> lock(mutex_); // 加锁以保证线程安全

    // 1. 尝试从缓存中查找合适大小的空闲块
    auto it = free_blocks_.find(aligned_size);
    if (it != free_blocks_.end() && !it->second.empty()) {
      // 找到了匹配大小的空闲块列表，并且列表不为空

      // 从列表末尾取出一个块（pop_back 比 vector 前端操作更高效）
      void* ptr = it->second.back();
      it->second.pop_back();

      // 将取出的块重新标记为活跃状态，记录其指针和大小
      active_allocations_[ptr] = aligned_size;

      // 返回分配的内存指针
      return ptr;
    }

    // 2. 缓存中没有合适的块，需要调用 cudaMalloc 分配新的 GPU 内存
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, aligned_size);

    if (err != cudaSuccess) {
      // cudaMalloc 分配失败
      std::cerr << "CudaMemoryPool 错误: cudaMalloc 分配失败，请求大小 "
                << aligned_size << " (原始请求 " << size
                << ")。错误: " << cudaGetErrorString(err) << std::endl;

      // 可选：尝试清理缓存并重试？ (这里未实现，直接返回失败)
      // 例如：可以调用 trim_internal() 或 trim_threshold_internal() 释放部分或全部缓存
      // trim_internal(); // 释放所有缓存
      // err = cudaMalloc(&ptr, aligned_size); // 再次尝试
      // if (err == cudaSuccess) { ... }

      return nullptr; // 分配失败，返回 nullptr
    }

    // 新内存分配成功
    // 记录这次新的活跃分配
    active_allocations_[ptr] = aligned_size;

    // 返回新分配的内存指针
    return ptr;
  }

  // 释放内存回内存池
  void free(void* ptr) {
    if (!ptr) {
      // 对空指针调用 free 是无操作
      return;
    }

    std::lock_guard<std::mutex> lock(mutex_); // 加锁以保证线程安全

    // 1. 检查这个指针是否是由本内存池分配并管理的活跃块
    auto it = active_allocations_.find(ptr);
    if (it == active_allocations_.end()) {
      // 如果在活跃列表中找不到这个指针，说明它不是由本池管理，或者已经被释放了。
      // 这通常是一个错误，或者是在程序退出时静态对象的析构顺序问题。
      std::cerr << "警告: 尝试释放一个不由 CudaMemoryPool 管理的指针 (" << ptr
                << ") 或已被释放的指针。" << std::endl;

      // 获取调用栈信息，帮助调试 (需要链接 -rdynamic 选项)
      // void* array[10];
      // size_t stack_size = backtrace(array, 10);
      // char** strings = backtrace_symbols(array, stack_size);
      // std::cerr << "调用栈:\n";
      // for (size_t i = 0; i < stack_size; i++) {
      //     std::cerr << strings[i] << std::endl;
      // }
      // ::free(strings); // 使用 C 库的 free 释放 backtrace_symbols 返回的内存

      return; // 不做任何操作
    }

    // 2. 获取该内存块记录的对齐后的大小
    size_t aligned_size = it->second;

    // 3. 将内存块放回对应大小的空闲块列表（缓存起来）
    // 使用 emplace_back 可能比 push_back 略优（如果 void* 需要构造的话，虽然这里不需要）
    free_blocks_[aligned_size].push_back(ptr);

    // 4. 从活跃分配映射中移除该指针，因为它现在是空闲状态
    active_allocations_.erase(it);
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

  // 获取内存池的统计信息结构体
  struct PoolStats {
    size_t total_cached_blocks = 0; // 空闲列表中总块数
    size_t total_cached_bytes = 0;  // 空闲列表中总字节数 (GPU内存)
    size_t active_allocations = 0;  // 当前活跃（未释放回池）的块数
    size_t active_bytes = 0;        // 当前活跃的总字节数 (GPU内存)
    size_t size_categories = 0;     // 空闲列表中有多少种不同的大小类别
  };

  // 获取当前的统计信息
  PoolStats getStats() const {
    std::lock_guard<std::mutex> lock(mutex_); // 加锁（因为要访问共享数据）

    PoolStats stats;
    stats.size_categories = free_blocks_.size();       // 获取空闲类别数量
    stats.active_allocations = active_allocations_.size(); // 获取活跃块数量

    // 遍历所有空闲块类别，累加缓存的块数和字节数
    for (const auto& [size, blocks] : free_blocks_) {
      stats.total_cached_blocks += blocks.size();
      stats.total_cached_bytes += size * blocks.size();
    }

    // 使用 std::accumulate 计算所有活跃分配的总字节数
    stats.active_bytes = std::accumulate(
        active_allocations_.begin(), active_allocations_.end(), 0ULL, // 使用 0ULL 确保是 size_t/unsigned long long 类型
        [](size_t sum, const auto& pair) { return sum + pair.second; }); // pair.second 是记录的大小

    return stats;
  }

 private:
  // 内部实现的 trim 函数，在已持有锁的情况下调用
  void trim_internal(size_t size = 0) {
    if (size == 0) {
      // size 为 0，表示释放所有缓存块
      for (auto& [block_size, blocks] : free_blocks_) {
        for (void* ptr : blocks) {
          cudaError_t err = cudaFree(ptr); // 释放 GPU 内存
           if (err != cudaSuccess) {
             std::cerr << "警告: CudaMemoryPool trim_internal cudaFree 失败 (大小: " << block_size << "): "
                       << cudaGetErrorString(err) << std::endl;
           }
        }
        // blocks.clear(); // 清空 vector (不需要，因为下面会 clear 整个 map)
      }
      free_blocks_.clear(); // 清空整个空闲块映射
    } else {
      // size 不为 0，表示只释放指定大小类别的缓存块
      size_t aligned_size = (size + 255) & ~255; // 计算对齐后的大小
      auto it = free_blocks_.find(aligned_size);
      if (it != free_blocks_.end()) {
        // 找到了该大小的类别
        for (void* ptr : it->second) {
          cudaError_t err = cudaFree(ptr); // 释放该类别下的所有 GPU 内存块
           if (err != cudaSuccess) {
             std::cerr << "警告: CudaMemoryPool trim_internal cudaFree 失败 (大小: " << aligned_size << "): "
                       << cudaGetErrorString(err) << std::endl;
           }
        }
        free_blocks_.erase(it); // 从 map 中移除该大小类别
      }
    }
  }

  // 内部实现的 trim_threshold 函数，在已持有锁的情况下调用
  void trim_threshold_internal(size_t max_blocks_per_size) {
    // 使用迭代器遍历 map，因为我们可能需要删除元素
    for (auto it = free_blocks_.begin(); it != free_blocks_.end(); /* no increment here */) {
      size_t size = it->first;
      std::vector<void*>& blocks = it->second;

      if (blocks.size() > max_blocks_per_size) {
        // 当前类别的缓存块数量超过阈值
        size_t blocks_to_release = blocks.size() - max_blocks_per_size; // 计算需要释放的数量

        // 从 vector 的末尾开始释放多余的块
        for (size_t i = 0; i < blocks_to_release; ++i) {
          // 检查 vector 是否为空（理论上不会，但防御性编程）
          if (blocks.empty()) break;

          void* ptr_to_free = blocks.back(); // 获取最后一个块的指针
          cudaError_t err = cudaFree(ptr_to_free); // 释放 GPU 内存
          if (err != cudaSuccess) {
            std::cerr << "警告: CudaMemoryPool trim_threshold cudaFree 失败 (大小: "
                      << size << "): " << cudaGetErrorString(err) << std::endl;
            // 即使释放失败，我们仍然从记录中移除它，避免无限循环或状态不一致
          }
          blocks.pop_back(); // 从 vector 中移除记录
        }
      }

      // 检查修剪后该类别是否变为空
      if (blocks.empty()) {
        // 如果 vector 空了，从 map 中移除这个大小类别，使用迭代器进行安全删除
        it = free_blocks_.erase(it); // erase 返回下一个有效迭代器
      } else {
        // 如果未删除，则手动递增迭代器
        ++it;
      }
    }
  }

  // 核心数据结构：
  // 存储空闲内存块的 map。键是 对齐后的内存块大小(size_t)，值是 存储该大小空闲块指针的 vector。
  std::unordered_map<size_t, std::vector<void*>> free_blocks_;

  // 存储当前活跃（已分配但未释放回池）内存块的 map。
  // 键是 内存块指针(void*)，值是 该内存块对应的对齐后的大小(size_t)。
  // 用于在 free 时查找块的大小并验证指针的有效性。
  std::unordered_map<void*, size_t> active_allocations_;

  // 互斥锁，用于保护对 free_blocks_ 和 active_allocations_ 的并发访问，确保线程安全。
  // mutable 关键字允许在 const 成员函数 (如 getStats) 中锁定互斥锁。
  mutable std::mutex mutex_;

  // C++11 的线程安全初始化标志，配合 std::call_once 使用。
  // 注意：这里在 CudaMemoryPool 类内部声明只是为了完整性，实际的单例标志在 GlobalCudaMemoryPool 中。
  // static std::once_flag init_flag_; // 不需要放在这里
};


// 提供一个全局单例访问点 (线程安全，最长生命周期)
class GlobalCudaMemoryPool {
 public:
  // 获取全局唯一的 CudaMemoryPool 实例引用
  static CudaMemoryPool& instance();

 private:
  // 私有构造/析构/拷贝/赋值，防止外部创建 GlobalCudaMemoryPool 的实例
  GlobalCudaMemoryPool() = default;
  ~GlobalCudaMemoryPool() = default; // 析构函数是默认的，但不会被调用
  GlobalCudaMemoryPool(const GlobalCudaMemoryPool&) = delete;
  GlobalCudaMemoryPool& operator=(const GlobalCudaMemoryPool&) = delete;
  GlobalCudaMemoryPool(GlobalCudaMemoryPool&&) = delete;
  GlobalCudaMemoryPool& operator=(GlobalCudaMemoryPool&&) = delete;

  // 指向全局单例 CudaMemoryPool 实例的指针
  static CudaMemoryPool* pool_instance_ptr;
  // 用于保证线程安全初始化的标志
  static std::once_flag init_flag_;
};