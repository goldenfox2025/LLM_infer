#pragma once

#include <cuda.h>  // <--- 引入CUDA Driver API以使用VMM
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

class CudaMemoryPool;

class CudaMemoryPool {
   public:
    CudaMemoryPool()
        : is_shutting_down_(false),
          is_prefill_mode_(false),
          is_prefill_phase_(false),
          prefill_buffer_(nullptr),
          prefill_buffer_size_(0),
          prefill_buffer_used_(0),
          prefill_buffer_committed_(0),
          prefill_max_size_(256 * 1024 * 1024),
          vmm_granularity_(0) {
        // 调用cuInit(0)确保Driver API已初始化, 多次调用是安全的。
        // 调用cudaFree(0)可初始化CUDA Runtime和上下文。
        cuInit(0);
        cudaError_t err = cudaFree(0);
        if (err != cudaSuccess && err != cudaErrorInvalidDevicePointer) {
            // 初始化检查失败，可能表示CUDA环境存在问题。
        }
    }

    ~CudaMemoryPool() {
        std::lock_guard<std::mutex> lock(mutex_);
        is_shutting_down_ = true;

        // 优先清理VMM资源
        if (is_prefill_mode_) {
            disable_prefill_mode_internal();
        }

        bool driver_available = is_cuda_driver_available();

        if (driver_available) {
            // 释放所有缓存的内存块
            for (auto& [size, blocks] : free_blocks_) {
                for (void* ptr : blocks) {
                    safe_cuda_free(ptr, driver_available);
                }
            }

            // 释放所有标签化内存块
            for (auto const& [tag, block_info] : tagged_memory_) {
                safe_cuda_free(block_info.ptr, driver_available);
            }

            // VMM prefill_buffer_的释放已由disable_prefill_mode_internal处理
            // 无需再调用 safe_cuda_free(prefill_buffer_, driver_available);
        }

        // 无论驱动状态如何，都清空所有跟踪记录
        free_blocks_.clear();
        tagged_memory_.clear();
        memory_tags_.clear();
        active_allocations_.clear();
        prefill_buffer_ = nullptr;
        prefill_buffer_size_ = 0;
        prefill_buffer_used_ = 0;
        prefill_buffer_committed_ = 0;
        vmm_chunks_.clear();
    }

    // 删除拷贝和移动操作，保证单例。
    CudaMemoryPool(const CudaMemoryPool&) = delete;
    CudaMemoryPool& operator=(const CudaMemoryPool&) = delete;
    CudaMemoryPool(CudaMemoryPool&&) = delete;
    CudaMemoryPool& operator=(CudaMemoryPool&&) = delete;

    // --- 公开接口 ---

    void* allocate(size_t size, bool is_prefill_request = false, const std::string& tag = "") {
        if (size == 0)
            return nullptr;
        if (!tag.empty()) {
            // 重定向到专用的标签分配函数。
            return allocate_tagged(tag, size, is_prefill_request);
        }

        std::lock_guard<std::mutex> lock(mutex_);
        if (is_shutting_down_)
            return nullptr;

        size_t aligned_size = (size + 255) & ~255;
        void* ptr = nullptr;

        // Prefill阶段的分配优先使用prefill缓冲区。
        bool use_prefill = is_prefill_request || is_prefill_phase_;
        if (use_prefill) {
            ptr = try_allocate_from_prefill_internal(aligned_size);
            if (ptr)
                return ptr;
        }

        // 尝试从缓存中寻找合适的内存块。
        ptr = try_allocate_from_cache_internal(aligned_size);
        if (ptr)
            return ptr;

        // 最后再分配新的内存块。
        return allocate_new_block_internal(aligned_size, "");
    }

    void free(void* ptr) {
        if (!ptr)
            return;

        std::lock_guard<std::mutex> lock(mutex_);
        if (is_shutting_down_) {
            // 关闭时，仅停止跟踪分配，由析构函数统一释放。
            active_allocations_.erase(ptr);
            return;
        }

        auto it_active = active_allocations_.find(ptr);
        if (it_active == active_allocations_.end()) {
            // 未找到指针，可能是重复释放或无效指针。
            return;
        }
        size_t aligned_size = it_active->second;

        // 检查指针是否属于标签内存块，若是则标记为非活动。
        auto it_tag = memory_tags_.find(ptr);
        if (it_tag != memory_tags_.end()) {
            tagged_memory_[it_tag->second].is_active = false;
            active_allocations_.erase(it_active);
            return;
        }

        // 检查指针是否来自prefill缓冲区。
        if (is_from_prefill_buffer_internal(ptr)) {
            free_from_prefill_buffer_internal(ptr, aligned_size);
            active_allocations_.erase(it_active);
            return;
        }

        // 对常规分配，进行缓存或直接释放。
        if (should_cache_block_internal(aligned_size)) {
            free_blocks_[aligned_size].push_back(ptr);
        } else {
            bool driver_ok = is_cuda_driver_available();
            safe_cuda_free(ptr, driver_ok);
        }

        active_allocations_.erase(it_active);
        perform_periodic_cleanup_internal();
    }

    void* allocate_tagged(const std::string& tag, size_t size, bool is_prefill_request = false) {
        if (tag.empty())
            return allocate(size, is_prefill_request);
        if (size == 0)
            return nullptr;

        std::lock_guard<std::mutex> lock(mutex_);
        if (is_shutting_down_)
            return nullptr;

        size_t aligned_size = (size + 255) & ~255;

        auto it = tagged_memory_.find(tag);
        if (it != tagged_memory_.end()) {
            TaggedBlockInfo& block_info = it->second;

            // 内存块已存在，检查是否可用。
            if (block_info.size >= aligned_size) {
                if (!block_info.is_active) {
                    block_info.is_active = true;
                    active_allocations_[block_info.ptr] = block_info.size;
                }
                return block_info.ptr;
            } else {
                // 现有内存块太小，释放它再分配新的。
                if (block_info.is_active) {
                    active_allocations_.erase(block_info.ptr);
                }
                bool driver_ok = is_cuda_driver_available();
                safe_cuda_free(block_info.ptr, driver_ok);
                memory_tags_.erase(block_info.ptr);
                tagged_memory_.erase(it);
            }
        }

        // 为标签分配一个新的内存块 (标签内存不使用prefill缓冲区)。
        return allocate_new_block_internal(aligned_size, tag);
    }

       void* get_tagged_memory(const std::string& tag) {
        if (tag.empty())
            return nullptr;
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = tagged_memory_.find(tag);
        return (it != tagged_memory_.end()) ? it->second.ptr : nullptr;
    }

    bool has_tag(const std::string& tag) {
        if (tag.empty())
            return false;
        std::lock_guard<std::mutex> lock(mutex_);
        return tagged_memory_.count(tag) > 0;
    }

    void trim(size_t size = 0) {
        std::lock_guard<std::mutex> lock(mutex_);
        trim_internal(size);
    }

    void trim_threshold(size_t max_blocks_per_size) {
        std::lock_guard<std::mutex> lock(mutex_);
        trim_threshold_internal(max_blocks_per_size);
    }

    bool enable_prefill_mode(size_t initial_size = 48 * 1024 * 1024, size_t max_size = 512 * 1024 * 1024) {
        std::lock_guard<std::mutex> lock(mutex_);
        return enable_prefill_mode_internal(initial_size, max_size);
    }

    void disable_prefill_mode() {
        std::lock_guard<std::mutex> lock(mutex_);
        disable_prefill_mode_internal();
    }

    void reset_prefill_buffer() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (is_prefill_mode_ && prefill_buffer_ != nullptr) {
            prefill_buffer_used_ = 0;
            // 注意：这不会从active_allocations_移除prefill分配。
            // 调用者需保证reset后不再使用这些指针。
        }
    }

    void set_prefill_phase(bool is_prefill) {
        std::lock_guard<std::mutex> lock(mutex_);
        is_prefill_phase_ = is_prefill;
    }

    void prepare_for_shutdown() {
        std::lock_guard<std::mutex> lock(mutex_);
        is_shutting_down_ = true;
        trim_internal(0);  // 清理所有缓存
        std::cerr << "CudaMemoryPool: 已准备安全关闭，将限制后续CUDA操作。" << std::endl;
    }

    bool is_shutting_down() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return is_shutting_down_;
    }

    // --- 统计信息 ---
    struct PoolStats {
        size_t total_cached_blocks = 0;
        size_t total_cached_bytes = 0;
        size_t active_allocations_count = 0;
        size_t active_bytes = 0;
        size_t size_categories_in_cache = 0;
        size_t tagged_allocations_count = 0;
        size_t tagged_active_count = 0;
        size_t tagged_total_bytes = 0;
        size_t prefill_buffer_reserved_bytes = 0;
        size_t prefill_buffer_committed_bytes = 0;
        size_t prefill_buffer_used_bytes = 0;
    };

    PoolStats getStats() const {
        std::lock_guard<std::mutex> lock(mutex_);
        PoolStats stats;

        stats.active_allocations_count = active_allocations_.size();
        stats.active_bytes = std::accumulate(active_allocations_.begin(), active_allocations_.end(), 0ULL,
                                             [](size_t sum, const auto& pair) { return sum + pair.second; });

        stats.size_categories_in_cache = free_blocks_.size();
        for (const auto& [size_val, blocks] : free_blocks_) {
            stats.total_cached_blocks += blocks.size();
            stats.total_cached_bytes += size_val * blocks.size();
        }

        stats.tagged_allocations_count = tagged_memory_.size();
        for (const auto& [tag, block_info] : tagged_memory_) {
            stats.tagged_total_bytes += block_info.size;
            if (block_info.is_active) {
                stats.tagged_active_count++;
            }
        }

        stats.prefill_buffer_reserved_bytes = prefill_buffer_size_;
        stats.prefill_buffer_committed_bytes = prefill_buffer_committed_;
        stats.prefill_buffer_used_bytes = prefill_buffer_used_;
        return stats;
    }

   private:
    // --- 内部数据结构 ---

    struct TaggedBlockInfo {
        void* ptr = nullptr;
        size_t size = 0;
        bool is_active = false;
    };

    // VMM块信息
    struct VmmChunk {
        CUmemGenericAllocationHandle handle;
        size_t size = 0;
    };

    // --- 内部状态变量 ---

    mutable std::mutex mutex_;
    bool is_shutting_down_;

    // 常规缓存状态
    std::unordered_map<size_t, std::vector<void*>> free_blocks_;
    std::unordered_map<void*, size_t> active_allocations_;

    // 标签内存状态
    std::map<std::string, TaggedBlockInfo> tagged_memory_;
    std::map<void*, std::string> memory_tags_;

    // Prefill 缓冲区状态 (VMM实现)
    bool is_prefill_mode_;
    bool is_prefill_phase_;
    void* prefill_buffer_;             // 指向VMM预留的虚拟地址空间(VA)的起始位置
    size_t prefill_buffer_size_;       // VMM预留的VA总大小
    size_t prefill_buffer_used_;       // 在VA空间中已分配出去的大小(碰撞指针)
    size_t prefill_buffer_committed_;  // 已映射到VA的物理内存大小
    size_t prefill_max_size_;
    size_t vmm_granularity_;            // VMM分配的粒度
    std::vector<VmmChunk> vmm_chunks_;  // 跟踪所有物理内存块

    // --- 内部辅助方法 ---

    void* try_allocate_from_prefill_internal(size_t aligned_size) {
        if (!is_prefill_mode_ || !prefill_buffer_) {
            return nullptr;
        }

        // 检查当前已提交的物理内存是否足够
        if (prefill_buffer_used_ + aligned_size <= prefill_buffer_committed_) {
            void* ptr = static_cast<char*>(prefill_buffer_) + prefill_buffer_used_;
            prefill_buffer_used_ += aligned_size;
            active_allocations_[ptr] = aligned_size;
            return ptr;
        }

        // 物理内存不足，尝试映射新的物理块进行扩容
        // 检查预留的虚拟地址空间是否还足够
        if (prefill_buffer_committed_ >= prefill_buffer_size_) {
            return nullptr;  // 虚拟地址空间已用完
        }

        // 计算需要扩容的大小，至少为请求大小，但通常更大以减少扩容次数
        size_t min_new_chunk_size = std::max(vmm_granularity_, aligned_size);
        size_t new_chunk_size = std::max(min_new_chunk_size * 2, (prefill_buffer_committed_ / 4));  // 扩容当前大小的25%
        new_chunk_size = (new_chunk_size + vmm_granularity_ - 1) & ~(vmm_granularity_ - 1);         // 对齐到粒度
        new_chunk_size =
            std::min(new_chunk_size, prefill_buffer_size_ - prefill_buffer_committed_);  // 不能超过剩余虚拟空间

        if (new_chunk_size < aligned_size)
            return nullptr;  // 即使扩容也无法满足

        // 创建新的物理内存块
        CUmemGenericAllocationHandle handle;
        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location = {CU_MEM_LOCATION_TYPE_DEVICE, 0};
        if (cuMemCreate(&handle, new_chunk_size, &prop, 0) != CUDA_SUCCESS) {
            return nullptr;
        }
       // 将新的物理块映射到虚拟地址空间的末尾
        CUdeviceptr va_ptr = reinterpret_cast<CUdeviceptr>(prefill_buffer_);
        if (cuMemMap(va_ptr + prefill_buffer_committed_, new_chunk_size, 0, handle, 0) != CUDA_SUCCESS) {
            cuMemRelease(handle);
            return nullptr;
        }

        // 设置内存访问权限
        CUmemAccessDesc accessDesc = {};
        accessDesc.location = prop.location;
        accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        if (cuMemSetAccess(va_ptr + prefill_buffer_committed_, new_chunk_size, &accessDesc, 1) != CUDA_SUCCESS) {
            cuMemUnmap(va_ptr + prefill_buffer_committed_, new_chunk_size);
            cuMemRelease(handle);
            return nullptr;
        }

        // 更新状态
        vmm_chunks_.push_back({handle, new_chunk_size});
        prefill_buffer_committed_ += new_chunk_size;

        // 扩容成功后，再次进行分配
        void* ptr = static_cast<char*>(prefill_buffer_) + prefill_buffer_used_;
        prefill_buffer_used_ += aligned_size;
        active_allocations_[ptr] = aligned_size;
        return ptr;
    }

    // ... [其它未改动的内部方法: try_allocate_from_cache_internal, allocate_new_block_internal, etc.]
    void* try_allocate_from_cache_internal(size_t aligned_size) {
        auto it = free_blocks_.find(aligned_size);
        if (it != free_blocks_.end() && !it->second.empty()) {
            void* ptr = it->second.back();
            it->second.pop_back();
            if (it->second.empty()) {
                free_blocks_.erase(it);
            }
            active_allocations_[ptr] = aligned_size;
            return ptr;
        }
        return nullptr;
    }

    void* allocate_new_block_internal(size_t aligned_size, const std::string& tag) {
        size_t free_memory = 0, total_memory = 0;
        if (cudaMemGetInfo(&free_memory, &total_memory) == cudaSuccess) {
            if (aligned_size > free_memory) {
                trim_threshold_internal(2);
                if (cudaMemGetInfo(&free_memory, &total_memory) == cudaSuccess && aligned_size > free_memory) {
                    trim_internal(0);
                }
            }
        }

        void* ptr = nullptr;
        if (cudaMalloc(&ptr, aligned_size) != cudaSuccess) {
            return nullptr;
        }

        active_allocations_[ptr] = aligned_size;

        if (!tag.empty()) {
            tagged_memory_[tag] = {ptr, aligned_size, true};
            memory_tags_[ptr] = tag;
        }
        return ptr;
    }

    bool is_from_prefill_buffer_internal(const void* ptr) const {
        if (!is_prefill_mode_ || !prefill_buffer_)
            return false;
        const char* p = static_cast<const char*>(ptr);
        const char* start = static_cast<const char*>(prefill_buffer_);
        // 使用预留的虚拟地址空间总大小来判断
        const char* end = start + prefill_buffer_size_;
        return p >= start && p < end;
    }

    void free_from_prefill_buffer_internal(void* ptr, size_t aligned_size) {
        // 此函数为空是设计如此。
        // VMM prefill缓冲区作为“竞技场”分配器工作。
        // 内存只在整个缓冲区被重置时(reset_prefill_buffer)或关闭时(disable_prefill_mode)才统一回收。
        // 单个内存的free操作仅在调用方free()中从active_allocations_移除记录。
    }

    bool should_cache_block_internal(size_t aligned_size) const {
        size_t max_blocks = 8;
        if (aligned_size >= 1024 * 1024)
            max_blocks = 2;  // >= 1MB
        else if (aligned_size >= 65536)
            max_blocks = 4;  // >= 64KB

        auto it = free_blocks_.find(aligned_size);
        if (it == free_blocks_.end())
            return true;
        return it->second.size() < max_blocks;
    }

    void perform_periodic_cleanup_internal() {
        size_t total_cached_blocks = 0;
        for (const auto& [size_key, blocks] : free_blocks_) {
            total_cached_blocks += blocks.size();
        }
        if (total_cached_blocks > 100) {
            trim_threshold_internal(4);
        }
    }

    void trim_internal(size_t size) {
        bool driver_ok = is_cuda_driver_available();
        if (size == 0) {
            for (auto& [block_size, blocks] : free_blocks_) {
                for (void* ptr : blocks) {
                    safe_cuda_free(ptr, driver_ok);
                }
            }
            free_blocks_.clear();
        } else {
            size_t aligned_size = (size + 255) & ~255;
            auto it = free_blocks_.find(aligned_size);
            if (it != free_blocks_.end()) {
                for (void* ptr : it->second) {
                    safe_cuda_free(ptr, driver_ok);
                }
                free_blocks_.erase(it);
            }
        }
    }

    void trim_threshold_internal(size_t max_blocks_per_size) {
        bool driver_ok = is_cuda_driver_available();
        for (auto it = free_blocks_.begin(); it != free_blocks_.end();) {
            while (it->second.size() > max_blocks_per_size) {
                void* ptr = it->second.back();
                it->second.pop_back();
                safe_cuda_free(ptr, driver_ok);
            }
            if (it->second.empty()) {
                it = free_blocks_.erase(it);
            } else {
                ++it;
            }
        }
    }

    bool enable_prefill_mode_internal(size_t initial_size, size_t max_size) {
        if (is_shutting_down_)
            return false;

        // 如果已开启，先关闭以释放旧资源
        if (is_prefill_mode_) {
            disable_prefill_mode_internal();
        }

        // 获取VMM分配粒度
        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location = {CU_MEM_LOCATION_TYPE_DEVICE, 0};
        cuMemGetAllocationGranularity(&vmm_granularity_, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
        if (vmm_granularity_ == 0)
            return false;

        prefill_max_size_ = (max_size + vmm_granularity_ - 1) & ~(vmm_granularity_ - 1);

        // 1. 预留虚拟地址空间(VA)
        CUdeviceptr ptr_d = 0;
        if (cuMemAddressReserve(&ptr_d, prefill_max_size_, 0, 0, 0) != CUDA_SUCCESS) {
            return false;
        }
        prefill_buffer_ = reinterpret_cast<void*>(ptr_d);
        prefill_buffer_size_ = prefill_max_size_;

        // 2. 分配并映射初始物理块
        size_t adjusted_initial_size = adjust_prefill_size_internal(initial_size);
        if (adjusted_initial_size > 0) {
            CUmemGenericAllocationHandle handle;
            if (cuMemCreate(&handle, adjusted_initial_size, &prop, 0) != CUDA_SUCCESS) {
                cuMemAddressFree(ptr_d, prefill_max_size_);
                prefill_buffer_ = nullptr;
                return false;
            }

            if (cuMemMap(ptr_d, adjusted_initial_size, 0, handle, 0) != CUDA_SUCCESS) {
                cuMemRelease(handle);
                cuMemAddressFree(ptr_d, prefill_max_size_);
                prefill_buffer_ = nullptr;
                return false;
            }

            CUmemAccessDesc accessDesc = {};
            accessDesc.location = prop.location;
            accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            cuMemSetAccess(ptr_d, adjusted_initial_size, &accessDesc, 1);

            vmm_chunks_.push_back({handle, adjusted_initial_size});
            prefill_buffer_committed_ = adjusted_initial_size;
        }

        prefill_buffer_used_ = 0;
        is_prefill_mode_ = true;
        std::cerr << "CudaMemoryPool: Prefill模式已开启(VMM)。预留: " << (prefill_buffer_size_ / (1024.0 * 1024.0))
                  << " MB, 初始提交: " << (prefill_buffer_committed_ / (1024.0 * 1024.0)) << " MB" << std::endl;
        return true;
    }

    void disable_prefill_mode_internal() {
        if (!is_prefill_mode_ || !prefill_buffer_)
            return;

        CUdeviceptr va_ptr = reinterpret_cast<CUdeviceptr>(prefill_buffer_);
        size_t offset = 0;

        // 取消映射并释放所有物理块
        for (const auto& chunk : vmm_chunks_) {
            cuMemUnmap(va_ptr + offset, chunk.size);
            cuMemRelease(chunk.handle);
            offset += chunk.size;
        }

        // 释放整个虚拟地址空间
        if (prefill_buffer_size_ > 0) {
            cuMemAddressFree(va_ptr, prefill_buffer_size_);
        }

        vmm_chunks_.clear();
        prefill_buffer_ = nullptr;
        prefill_buffer_size_ = 0;
        prefill_buffer_used_ = 0;
        prefill_buffer_committed_ = 0;
        is_prefill_mode_ = false;
        std::cerr << "CudaMemoryPool: Prefill模式已关闭。" << std::endl;
    }

    size_t adjust_prefill_size_internal(size_t requested_size) {
        if (vmm_granularity_ == 0)
            return 0;
        size_t free_memory = 0, total_memory = 0;
        if (cudaMemGetInfo(&free_memory, &total_memory) == cudaSuccess) {
            // 预留20%的空闲显存
            if (requested_size > free_memory * 0.8) {
                requested_size = static_cast<size_t>(free_memory * 0.7);
            }
        }
        // 将大小向上对齐到VMM粒度
        return (std::max(requested_size, vmm_granularity_) + vmm_granularity_ - 1) & ~(vmm_granularity_ - 1);
    }

   
    bool is_cuda_driver_available() {
        cudaError_t err = cudaFree(0);
        return err == cudaSuccess || err == cudaErrorInvalidDevicePointer;
    }

    void safe_cuda_free(void* ptr, bool& driver_available_flag) {
        if (!ptr || !driver_available_flag)
            return;

        cudaError_t err = cudaFree(ptr);
        if (err != cudaSuccess && err == cudaErrorCudartUnloading) {
            driver_available_flag = false;
        }
    }
};

// ==================================================================
// 全局单例包装器
// ==================================================================

class GlobalCudaMemoryPool {
   public:
    static CudaMemoryPool& instance() {
        std::call_once(init_flag_, []() {
            std::lock_guard<std::mutex> lock(init_mutex_);
            if (!pool_instance_ptr) {
                pool_instance_ptr = new CudaMemoryPool();
            }
        });
        return *pool_instance_ptr;
    }

    // --- 静态接口转发 ---

    static void* allocate(size_t size, bool is_prefill_request = false, const std::string& tag = "") {
        return instance().allocate(size, is_prefill_request, tag);
    }

    static void free(void* ptr) {
        instance().free(ptr);
    }

    static bool enable_prefill_mode(size_t initial_size = 48 * 1024 * 1024, size_t max_size = 512 * 1024 * 1024) {
        return instance().enable_prefill_mode(initial_size, max_size);
    }

    static void disable_prefill_mode() {
        instance().disable_prefill_mode();
    }

    static void reset_prefill_buffer() {
        instance().reset_prefill_buffer();
    }

    static void set_prefill_phase(bool is_prefill) {
        instance().set_prefill_phase(is_prefill);
    }

    static void* allocate_tagged(const std::string& tag, size_t size, bool is_prefill_request = false) {
        return instance().allocate_tagged(tag, size, is_prefill_request);
    }

    static void* get_tagged_memory(const std::string& tag) {
        return instance().get_tagged_memory(tag);
    }

    static bool has_tag(const std::string& tag) {
        return instance().has_tag(tag);
    }

    static void prepare_for_shutdown() {
        std::lock_guard<std::mutex> lock(init_mutex_);
        if (pool_instance_ptr != nullptr) {
            pool_instance_ptr->prepare_for_shutdown();
        }
    }

    static bool is_shutting_down() {
        std::lock_guard<std::mutex> lock(init_mutex_);
        if (pool_instance_ptr != nullptr) {
            return pool_instance_ptr->is_shutting_down();
        }
        return false;
    }

    static void explicitly_delete_pool_instance() {
        std::lock_guard<std::mutex> lock(init_mutex_);
        if (pool_instance_ptr) {
            delete pool_instance_ptr;
            pool_instance_ptr = nullptr;
        }
    }

   private:
    GlobalCudaMemoryPool() = default;
    ~GlobalCudaMemoryPool() = default;
    GlobalCudaMemoryPool(const GlobalCudaMemoryPool&) = delete;
    GlobalCudaMemoryPool& operator=(const GlobalCudaMemoryPool&) = delete;
    GlobalCudaMemoryPool(GlobalCudaMemoryPool&&) = delete;
    GlobalCudaMemoryPool& operator=(GlobalCudaMemoryPool&&) = delete;

    static CudaMemoryPool* pool_instance_ptr;
    static std::once_flag init_flag_;
    static std::mutex init_mutex_;
};