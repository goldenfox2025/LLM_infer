// // #pragma once
// // #include <cuda_runtime.h>
// // #include <algorithm>
// // #include <iostream>
// // #include <mutex>
// // #include <stdexcept>
// // #include <string>
// // #include <vector>
// // #include <unordered_map>
// // #include <unordered_set>  // Add this
// // #include <list>
// // class CudaMemoryPool {
// // public:
// //   CudaMemoryPool() {
// //     // 确保CUDA上下文已初始化
// //     cudaFree(0);

// //     // 初始化伙伴系统参数
// //     min_block_size_ = 256;  // 最小块大小256字节
// //     max_block_size_ = 1 << 30;  // 最大块1GB
// //     max_level_ = 0;
// //     size_t size = min_block_size_;

// //     // 计算最大级别
// //     while (size < max_block_size_) {
// //       size *= 2;
// //       max_level_++;
// //     }

// //     // 初始化每个级别的空闲链表
// //     free_lists_.resize(max_level_ + 1);
// //   }

// //   ~CudaMemoryPool() {
// //     try {
// //       std::lock_guard<std::mutex> lock(mutex_);
// //       // 释放所有分配的CUDA内存
// //       for (auto& pair : allocated_blocks_) {
// //         cudaFree(pair.first);
// //       }
// //     } catch (const std::exception& e) {
// //       std::cerr << "Exception in ~CudaMemoryPool: " << e.what() <<
// std::endl;
// //     }
// //   }

// //   void* allocate(size_t size) {
// //     if (size == 0) return nullptr;

// //     std::lock_guard<std::mutex> lock(mutex_);

// //     // 计算所需块的大小级别
// //     size_t padded_size = std::max(size, min_block_size_);
// //     size_t block_size = min_block_size_;
// //     int level = 0;

// //     while (block_size < padded_size) {
// //       block_size *= 2;
// //       level++;

// //       if (level > max_level_) {
// //         throw std::runtime_error("请求内存超过最大块大小");
// //       }
// //     }

// //     // 尝试找到或分割一个合适的块
// //     void* ptr = allocate_block(level);
// //     if (ptr) {
// //       // 记录分配的块
// //       allocated_blocks_[ptr] = {level, size};
// //       return ptr;
// //     }

// //     // 无法从伙伴系统获取内存，直接向CUDA申请
// //     void* new_ptr = nullptr;
// //     cudaError_t err = cudaMalloc(&new_ptr, size);
// //     if (err != cudaSuccess) {
// //       throw std::runtime_error("CUDA内存分配错误: " +
// //       std::string(cudaGetErrorString(err)));
// //     }

// //     // 记录为非伙伴系统块
// //     direct_allocations_[new_ptr] = size;
// //     return new_ptr;
// //   }

// //   void free(void* ptr) {
// //     if (!ptr) return;

// //     std::lock_guard<std::mutex> lock(mutex_);

// //     // 检查是否是伙伴系统分配的块
// //     auto it = allocated_blocks_.find(ptr);
// //     if (it != allocated_blocks_.end()) {
// //       // 释放伙伴系统块
// //       int level = it->second.level;
// //       allocated_blocks_.erase(it);
// //       free_block(ptr, level);
// //       return;
// //     }

// //     // 检查是否是直接分配的块
// //     auto direct_it = direct_allocations_.find(ptr);
// //     if (direct_it != direct_allocations_.end()) {
// //       // 直接释放CUDA内存
// //       cudaFree(ptr);
// //       direct_allocations_.erase(direct_it);
// //       return;
// //     }

// //     // 不是我们管理的，忽略
// //   }

// //   size_t getAllocatedSize() const {
// //     std::lock_guard<std::mutex> lock(mutex_);
// //     size_t total = 0;

// //     // 计算伙伴系统分配的内存
// //     for (const auto& pair : allocated_blocks_) {
// //       total += pair.second.size;
// //     }

// //     // 计算直接分配的内存
// //     for (const auto& pair : direct_allocations_) {
// //       total += pair.second;
// //     }

// //     return total;
// //   }

// // private:
// //   struct BlockInfo {
// //     int level;
// //     size_t size;
// //   };

// //   struct Block {
// //     void* ptr;
// //     Block* buddy;

// //     Block(void* p) : ptr(p), buddy(nullptr) {}
// //   };

// //   // 从指定级别分配块
// //   void* allocate_block(int level) {
// //     // 如果当前级别有空闲块，直接使用
// //     if (!free_lists_[level].empty()) {
// //       Block* block = free_lists_[level].front();
// //       free_lists_[level].pop_front();
// //       void* ptr = block->ptr;
// //       delete block;
// //       return ptr;
// //     }

// //     // 当前级别没有空闲块，尝试从更高级别分割
// //     if (level < max_level_) {
// //       void* larger_ptr = allocate_block(level + 1);
// //       if (larger_ptr) {
// //         // 分割获得的更大块
// //         size_t block_size = get_block_size(level);
// //         void* buddy_ptr = static_cast<char*>(larger_ptr) + block_size;

// //         // 将伙伴块添加到空闲列表
// //         Block* buddy = new Block(buddy_ptr);
// //         free_lists_[level].push_back(buddy);
// //         return larger_ptr;
// //       }
// //     }

// //     // 如果没有更大的块可以分割，则尝试直接分配
// //     size_t block_size = get_block_size(level);
// //     void* new_ptr = nullptr;
// //     cudaError_t err = cudaMalloc(&new_ptr, block_size);
// //     if (err != cudaSuccess) {
// //       return nullptr;
// //     }

// //     // 记录为系统管理的块
// //     system_blocks_.insert(new_ptr);
// //     return new_ptr;
// //   }

// //   // 释放指定级别的块
// //   void free_block(void* ptr, int level) {
// //     Block* block = new Block(ptr);

// //     // 尝试合并伙伴块
// //     bool merged = try_merge(block, level);

// //     if (!merged) {
// //       // 未合并，将块添加到相应级别的空闲列表
// //       free_lists_[level].push_back(block);
// //     }
// //   }

// //   // 尝试合并伙伴块
// //   bool try_merge(Block* block, int level) {
// //     if (level >= max_level_) {
// //       return false;
// //     }

// //     // 计算伙伴块地址
// //     size_t block_size = get_block_size(level);
// //     uintptr_t addr = reinterpret_cast<uintptr_t>(block->ptr);
// //     uintptr_t buddy_addr = addr ^ block_size;
// //     void* buddy_ptr = reinterpret_cast<void*>(buddy_addr);

// //     // 查找伙伴块
// //     auto it = std::find_if(free_lists_[level].begin(),
// //     free_lists_[level].end(),
// //                           [buddy_ptr](const Block* b) { return b->ptr ==
// //                           buddy_ptr; });

// //     if (it != free_lists_[level].end()) {
// //       // 找到伙伴块，合并
// //       Block* buddy = *it;
// //       free_lists_[level].erase(it);

// //       // 合并后的块地址是两个块中较小的地址
// //       void* merged_ptr = (addr < buddy_addr) ? block->ptr : buddy_ptr;
// //       delete block;
// //       delete buddy;

// //       // 将合并后的块添加到更高级别
// //       Block* merged_block = new Block(merged_ptr);

// //       // 继续尝试向上合并
// //       bool further_merged = try_merge(merged_block, level + 1);
// //       if (!further_merged) {
// //         free_lists_[level + 1].push_back(merged_block);
// //       }

// //       return true;
// //     }

// //     return false;
// //   }

// //   // 获取指定级别的块大小
// //   size_t get_block_size(int level) const {
// //     return min_block_size_ << level;
// //   }

// //   // 系统参数
// //   size_t min_block_size_;
// //   size_t max_block_size_;
// //   int max_level_;

// //   // 数据结构
// //   std::vector<std::list<Block*>> free_lists_;  // 每个级别的空闲块链表
// //   std::unordered_map<void*, BlockInfo> allocated_blocks_;  //
// //   已分配的伙伴系统块 std::unordered_map<void*, size_t>
// direct_allocations_;
// //   // 直接分配的块 std::unordered_set<void*> system_blocks_;  //
// //   系统管理的全部块

// //   mutable std::mutex mutex_;
// // };

// #pragma once

// #include <cuda_runtime.h>

// #include <algorithm>
// #include <cmath>
// #include <iostream>
// #include <mutex>
// #include <shared_mutex>
// #include <unordered_map>
// #include <unordered_set>
// #include <vector>

// class CudaMemoryPool {
//  public:
//   CudaMemoryPool(size_t initial_pool_size = 1 << 28) {  // 默认256MB
//     // 确保CUDA上下文已初始化
//     cudaFree(0);

//     // 配置池参数
//     min_block_size_ = 256;  // 256字节对齐
//     max_block_size_ =
//         std::min(initial_pool_size, static_cast<size_t>(1 << 30));  //
//         最大1GB
//     max_level_ = static_cast<int>(std::log2(max_block_size_ /
//     min_block_size_));

//     // 初始化空闲列表并预留空间
//     free_lists_.resize(max_level_ + 1);
//     for (auto& list : free_lists_) {
//       list.reserve(8);  // 合理的初始容量
//     }

//     // 创建每个级别的锁 - 必须使用指针因为互斥锁不可复制
//     level_locks_.reserve(max_level_ + 1);
//     for (int i = 0; i <= max_level_; i++) {
//       level_locks_.push_back(std::make_unique<std::mutex>());
//     }

//     // 分配初始池
//     void* pool_ptr = nullptr;
//     cudaError_t err = cudaMalloc(&pool_ptr, max_block_size_);
//     if (err != cudaSuccess) {
//       std::cerr << "CUDA内存池初始化失败: " << cudaGetErrorString(err)
//                 << std::endl;
//       return;  // 池将在直通模式下运行（直接使用cudaMalloc）
//     }

//     // 添加到系统块并初始化最大空闲块
//     {
//       std::lock_guard<std::shared_mutex> write_lock(pool_mutex_);
//       system_blocks_.insert(pool_ptr);
//       base_ptr_ = pool_ptr;

//       // 将整个池添加到最大空闲列表
//       MemoryBlock block{pool_ptr, max_block_size_};
//       std::lock_guard<std::mutex> level_lock(*level_locks_[max_level_]);
//       free_lists_[max_level_].push_back(block);
//     }

//     initialized_ = true;
//   }

//   ~CudaMemoryPool() {
//     std::lock_guard<std::shared_mutex> write_lock(pool_mutex_);

//     // 释放所有分配的块
//     for (void* ptr : system_blocks_) {
//       cudaFree(ptr);
//     }

//     // 清除映射，以防止在静态析构顺序问题中使用后释放
//     allocated_blocks_.clear();
//     direct_allocations_.clear();
//     system_blocks_.clear();
//     free_lists_.clear();
//   }

//   void* allocate(size_t size) {
//     if (size == 0) return nullptr;

//     // 如果池未初始化，回退到直接分配
//     if (!initialized_) {
//       void* ptr = nullptr;
//       cudaError_t err = cudaMalloc(&ptr, size);
//       if (err != cudaSuccess) {
//         std::cerr << "直接CUDA分配失败: " << cudaGetErrorString(err)
//                   << std::endl;
//         return nullptr;
//       }

//       if (ptr) {
//         std::lock_guard<std::shared_mutex> write_lock(pool_mutex_);
//         direct_allocations_.insert(ptr);
//       }
//       return ptr;
//     }

//     // 向上取整到最小块大小
//     size_t aligned_size = (size + min_block_size_ - 1) & ~(min_block_size_ -
//     1); int target_level = level_from_size(aligned_size);

//     // 尝试从目标级别向上查找合适的块
//     for (int level = target_level; level <= max_level_; ++level) {
//       std::lock_guard<std::mutex> level_lock(*level_locks_[level]);

//       auto& free_list = free_lists_[level];
//       if (!free_list.empty()) {
//         // 选择最后一个块（比从前面弹出更快）
//         MemoryBlock block = free_list.back();
//         free_list.pop_back();

//         // 注册分配
//         {
//           std::lock_guard<std::shared_mutex> write_lock(pool_mutex_);
//           allocated_blocks_[block.ptr] = {target_level, aligned_size};
//         }

//         // 如果需要，分割块
//         if (level > target_level) {
//           split_block(block, level, target_level);
//         }

//         return block.ptr;
//       }
//     }

//     // 在池中未找到合适的块，使用直接分配
//     void* ptr = nullptr;
//     cudaError_t err = cudaMalloc(&ptr, aligned_size);
//     if (err != cudaSuccess) {
//       std::cerr << "直接CUDA分配失败: " << cudaGetErrorString(err) <<
//       std::endl; return nullptr;  // 分配失败
//     }

//     // 跟踪直接分配
//     {
//       std::lock_guard<std::shared_mutex> write_lock(pool_mutex_);
//       direct_allocations_.insert(ptr);
//     }

//     return ptr;
//   }

//   void free(void* ptr) {
//     if (!ptr) return;

//     // 检查这是否是直接分配
//     {
//       std::shared_lock<std::shared_mutex> read_lock(pool_mutex_);
//       if (direct_allocations_.find(ptr) != direct_allocations_.end()) {
//         read_lock.unlock();
//         std::lock_guard<std::shared_mutex> write_lock(pool_mutex_);
//         direct_allocations_.erase(ptr);
//         cudaFree(ptr);
//         return;
//       }

//       // 检查这是否是池分配
//       auto it = allocated_blocks_.find(ptr);
//       if (it == allocated_blocks_.end()) {
//         return;  // 不是我们的指针
//       }

//       // 在释放锁之前获取块信息
//       int level = it->second.level;
//       size_t size = get_block_size(level);

//       // 从已分配映射中删除
//       read_lock.unlock();
//       {
//         std::lock_guard<std::shared_mutex> write_lock(pool_mutex_);
//         allocated_blocks_.erase(ptr);
//       }

//       // 创建块用于合并
//       MemoryBlock block{ptr, size};

//       // 尝试与伙伴合并
//       merge_block(block, level);
//     }
//   }

//   // 获取池使用统计信息
//   struct PoolStats {
//     size_t total_pool_size;
//     size_t allocated_pool_memory;
//     size_t free_pool_memory;
//     size_t direct_allocations;
//     size_t fragmentation_percent;
//   };

//   PoolStats getStats() const {
//     std::shared_lock<std::shared_mutex> read_lock(pool_mutex_);

//     size_t allocated = 0;
//     for (const auto& [ptr, info] : allocated_blocks_) {
//       allocated += get_block_size(info.level);
//     }

//     size_t free_memory = 0;
//     for (int level = 0; level <= max_level_; ++level) {
//       free_memory += free_lists_[level].size() * get_block_size(level);
//     }

//     size_t direct_allocs = direct_allocations_.size();
//     size_t total = max_block_size_ * system_blocks_.size();

//     // 计算碎片化（越高 = 越碎片化）
//     size_t frag_percent = 0;
//     if (free_memory > 0) {
//       size_t ideal_free_blocks = 1;
//       size_t actual_free_blocks = 0;
//       for (int level = 0; level <= max_level_; ++level) {
//         actual_free_blocks += free_lists_[level].size();
//       }
//       frag_percent = actual_free_blocks > 0
//                          ? (100 * (actual_free_blocks - ideal_free_blocks)) /
//                                actual_free_blocks
//                          : 0;
//     }

//     return {.total_pool_size = total,
//             .allocated_pool_memory = allocated,
//             .free_pool_memory = free_memory,
//             .direct_allocations = direct_allocs,
//             .fragmentation_percent = frag_percent};
//   }

//  private:
//   // 块信息
//   struct BlockInfo {
//     int level;
//     size_t size;
//   };

//   // 内存块结构体
//   struct MemoryBlock {
//     void* ptr;
//     size_t size;

//     // 计算伙伴地址
//     void* buddy_address() const {
//       uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
//       return reinterpret_cast<void*>(addr ^ size);
//     }
//   };

//   // 核心数据结构
//   std::vector<std::vector<MemoryBlock>> free_lists_;
//   std::vector<std::unique_ptr<std::mutex>>
//       level_locks_;  // 使用指针避免互斥锁复制
//   std::unordered_map<void*, BlockInfo> allocated_blocks_;
//   std::unordered_set<void*> direct_allocations_;
//   std::unordered_set<void*> system_blocks_;

//   // 使用共享互斥锁允许并发读取
//   mutable std::shared_mutex pool_mutex_;

//   // 池参数
//   void* base_ptr_ = nullptr;
//   size_t min_block_size_;
//   size_t max_block_size_;
//   int max_level_;
//   bool initialized_ = false;

//   // 根据大小计算级别
//   int level_from_size(size_t size) const {
//     if (size <= min_block_size_) return 0;
//     return static_cast<int>(
//         std::ceil(std::log2(static_cast<double>(size) / min_block_size_)));
//   }

//   // 获取一个级别的块大小
//   size_t get_block_size(int level) const { return min_block_size_ << level; }

//   // 将块分割到目标级别
//   void split_block(MemoryBlock block, int current_level, int target_level) {
//     for (int level = current_level - 1; level >= target_level; --level) {
//       size_t half_size = get_block_size(level);
//       void* buddy_ptr = static_cast<char*>(block.ptr) + half_size;

//       // 将伙伴添加到空闲列表
//       MemoryBlock buddy{buddy_ptr, half_size};
//       {
//         std::lock_guard<std::mutex> level_lock(*level_locks_[level]);
//         free_lists_[level].push_back(buddy);
//       }

//       // 更新当前块
//       block.size = half_size;
//     }
//   }

//   // 尝试将块与其伙伴合并
//   void merge_block(const MemoryBlock& block, int level) {
//     MemoryBlock current = block;

//     while (level < max_level_) {
//       // 获取当前块的伙伴地址
//       size_t block_size = get_block_size(level);
//       uintptr_t current_addr = reinterpret_cast<uintptr_t>(current.ptr);

//       // 确保当前块地址正确对齐
//       if ((current_addr % block_size) != 0) {
//         break;  // 地址未正确对齐，不能合并
//       }

//       // 计算伙伴地址 (XOR 使用块大小而不是当前块大小)
//       uintptr_t buddy_addr = current_addr ^ block_size;
//       void* buddy_ptr = reinterpret_cast<void*>(buddy_addr);

//       // 确保伙伴在内存池范围内
//       uintptr_t base_addr = reinterpret_cast<uintptr_t>(base_ptr_);
//       uintptr_t pool_end = base_addr + max_block_size_;
//       if (buddy_addr < base_addr || buddy_addr >= pool_end) {
//         break;  // 伙伴不在内存池范围内
//       }

//       // 检查伙伴是否在空闲列表中
//       bool found = false;
//       {
//         std::lock_guard<std::mutex> level_lock(*level_locks_[level]);
//         auto& free_list = free_lists_[level];

//         // 寻找空闲列表中的伙伴
//         auto it = std::find_if(
//             free_list.begin(), free_list.end(),
//             [buddy_ptr](const MemoryBlock& b) { return b.ptr == buddy_ptr;
//             });

//         if (it != free_list.end()) {
//           free_list.erase(it);
//           found = true;
//         }
//       }

//       if (!found) {
//         // 伙伴不可用，将当前块添加到空闲列表后退出
//         std::lock_guard<std::mutex> level_lock(*level_locks_[level]);
//         free_lists_[level].push_back(current);
//         return;
//       }

//       // 合并：使用较低地址作为合并后的块地址
//       current.ptr = (buddy_addr > current_addr) ? current.ptr : buddy_ptr;
//       current.size = block_size * 2;
//       level++;
//     }

//     // 将最终块添加到适当的空闲列表
//     std::lock_guard<std::mutex> level_lock(*level_locks_[level]);
//     free_lists_[level].push_back(current);
//   }
// };
#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <iostream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>  // Add this
#include <vector>
class CudaMemoryPool {
 public:
  CudaMemoryPool() {
    // 确保CUDA上下文已初始化
    cudaFree(0);
  }

  ~CudaMemoryPool() {
    std::lock_guard<std::mutex> lock(mutex_);

    // 释放所有大小类别中的内存块
    for (auto& [size, blocks] : free_blocks_) {
      for (void* ptr : blocks) {
        cudaFree(ptr);
      }
    }

    // 释放所有直接分配的内存
    for (void* ptr : direct_allocations_) {
      cudaFree(ptr);
    }

    // 清空容器
    free_blocks_.clear();
    direct_allocations_.clear();
  }

  void* allocate(size_t size) {
    if (size == 0) return nullptr;

    // 基本对齐到256字节
    size_t aligned_size = (size + 255) & ~255;

    std::lock_guard<std::mutex> lock(mutex_);

    // 检查是否有该大小的空闲块
    auto it = free_blocks_.find(aligned_size);
    if (it != free_blocks_.end() && !it->second.empty()) {
      // 有可用的缓存块，取用最后一个（避免vector前端操作）
      void* ptr = it->second.back();
      it->second.pop_back();
      return ptr;
    }

    // 没有缓存块，直接分配新内存
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, aligned_size);
    if (err != cudaSuccess) {
      std::cerr << "CUDA分配失败: " << cudaGetErrorString(err) << std::endl;
      return nullptr;
    }

    // 记录此次分配
    direct_allocations_.insert(ptr);
    return ptr;
  }

  void free(void* ptr) {
    if (!ptr) return;

    std::lock_guard<std::mutex> lock(mutex_);

    // 查找此指针是否是我们之前分配的
    auto it = direct_allocations_.find(ptr);
    if (it == direct_allocations_.end()) {
      // 不是我们分配的内存，忽略
      return;
    }

    // 获取分配的大小（通过查询CUDA）
    size_t size = 0;
    cudaError_t err = cudaMemGetInfo(nullptr, &size);
    if (err != cudaSuccess) {
      // 如果无法获取大小信息，直接释放
      cudaFree(ptr);
      direct_allocations_.erase(it);
      return;
    }

    // 将内存块移到空闲列表
    size_t aligned_size = (size + 255) & ~255;
    free_blocks_[aligned_size].push_back(ptr);
    direct_allocations_.erase(it);
  }

  // 手动释放特定大小类别的缓存块
  void trim(size_t size = 0) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (size == 0) {
      // 释放所有缓存块
      for (auto& [block_size, blocks] : free_blocks_) {
        for (void* ptr : blocks) {
          cudaFree(ptr);
        }
        blocks.clear();
      }
    } else {
      // 释放特定大小的缓存块
      size_t aligned_size = (size + 255) & ~255;
      auto it = free_blocks_.find(aligned_size);
      if (it != free_blocks_.end()) {
        for (void* ptr : it->second) {
          cudaFree(ptr);
        }
        it->second.clear();
      }
    }
  }

  // 释放超过指定数量的缓存块
  void trim_threshold(size_t max_blocks_per_size) {
    std::lock_guard<std::mutex> lock(mutex_);

    for (auto& [size, blocks] : free_blocks_) {
      if (blocks.size() > max_blocks_per_size) {
        // 释放超过阈值的块
        size_t to_release = blocks.size() - max_blocks_per_size;
        for (size_t i = 0; i < to_release; i++) {
          cudaFree(blocks.back());
          blocks.pop_back();
        }
      }
    }
  }

  // 获取池统计信息
  struct PoolStats {
    size_t total_cached_blocks;
    size_t total_cached_bytes;
    size_t active_allocations;
    size_t size_categories;
  };

  PoolStats getStats() const {
    std::lock_guard<std::mutex> lock(mutex_);

    size_t total_blocks = 0;
    size_t total_bytes = 0;

    for (const auto& [size, blocks] : free_blocks_) {
      total_blocks += blocks.size();
      total_bytes += size * blocks.size();
    }

    return {.total_cached_blocks = total_blocks,
            .total_cached_bytes = total_bytes,
            .active_allocations = direct_allocations_.size(),
            .size_categories = free_blocks_.size()};
  }

 private:
  // 核心数据结构：按大小组织的空闲块向量
  std::unordered_map<size_t, std::vector<void*>> free_blocks_;

  // 当前活跃分配的集合
  std::unordered_set<void*> direct_allocations_;

  // 同步锁
  mutable std::mutex mutex_;
};