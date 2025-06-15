#pragma once

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
          prefill_max_size_(256 * 1024 * 1024) {
        cudaError_t err = cudaFree(0);
        if (err != cudaSuccess && err != cudaErrorInvalidDevicePointer) {
            // Initialization check failed
        }
    }

    ~CudaMemoryPool() {
        std::lock_guard<std::mutex> lock(mutex_);
        is_shutting_down_ = true;

        cudaError_t driver_status = cudaFree(0);
        bool driver_available = (driver_status == cudaSuccess || driver_status == cudaErrorInvalidDevicePointer);

        if (driver_available) {
            for (auto& [size, blocks] : free_blocks_) {
                for (void* ptr : blocks) {
                    cudaError_t err = cudaFree(ptr);
                    if (err != cudaSuccess) {
                        if (err == cudaErrorCudartUnloading) {
                            driver_available = false;
                            break;
                        }
                    }
                }
                if (!driver_available)
                    break;
            }
            free_blocks_.clear();

            if (driver_available && prefill_buffer_ != nullptr) {
                cudaError_t err = cudaFree(prefill_buffer_);
                if (err != cudaSuccess) {
                    if (err == cudaErrorCudartUnloading)
                        driver_available = false;
                } else {
                    prefill_buffer_ = nullptr;
                    prefill_buffer_size_ = 0;
                    prefill_buffer_used_ = 0;
                }
            }

            if (driver_available) {
                for (auto const& [tag_key, ptr_val] : tagged_memory_) {
                    cudaError_t err = cudaFree(ptr_val);
                    if (err != cudaSuccess) {
                        if (err == cudaErrorCudartUnloading) {
                            driver_available = false;
                            break;
                        }
                    }
                }
            }
            tagged_memory_.clear();
            memory_tags_.clear();
            tagged_memory_sizes_.clear();
            tagged_memory_active_.clear();

        } else {
            free_blocks_.clear();
            tagged_memory_.clear();
            memory_tags_.clear();
            tagged_memory_sizes_.clear();
            tagged_memory_active_.clear();
            prefill_buffer_ = nullptr;
            prefill_buffer_size_ = 0;
            prefill_buffer_used_ = 0;
        }

        if (!active_allocations_.empty()) {
            size_t active_bytes = 0;
            for (auto const& [ptr, size_val] : active_allocations_) {
                active_bytes += size_val;
            }
            active_allocations_.clear();
        }
    }

    CudaMemoryPool(const CudaMemoryPool&) = delete;
    CudaMemoryPool& operator=(const CudaMemoryPool&) = delete;
    CudaMemoryPool(CudaMemoryPool&&) = delete;
    CudaMemoryPool& operator=(CudaMemoryPool&&) = delete;

    void* allocate(size_t size, bool is_prefill_request = false, const std::string& tag = "") {
        if (size == 0)
            return nullptr;

        size_t aligned_size = (size + 255) & ~255;
        std::lock_guard<std::mutex> lock(mutex_);

        if (is_shutting_down_) {
            return nullptr;
        }

        bool actual_is_prefill_request = is_prefill_request;
        if (!actual_is_prefill_request && is_prefill_phase_) {
            actual_is_prefill_request = true;
        }

        void* ptr = nullptr;

        if (!tag.empty()) {
            if (actual_is_prefill_request) {
                ptr = try_allocate_from_prefill(size, aligned_size, true);
                if (ptr) {
                    tagged_memory_[tag] = ptr;
                    memory_tags_[ptr] = tag;
                    tagged_memory_sizes_[tag] = aligned_size;
                    tagged_memory_active_[tag] = true;
                    return ptr;
                }
            }
            return allocate_new_block(size, aligned_size, tag);
        } else {
            if (actual_is_prefill_request) {
                bool has_active_tagged_memory = false;
                for (const auto& [tag_name, is_active] : tagged_memory_active_) {
                    if (is_active) {
                        has_active_tagged_memory = true;
                        break;
                    }
                }
                
                if (!has_active_tagged_memory) {
                    ptr = try_allocate_from_prefill(size, aligned_size, true);
                }
            }
            if (ptr)
                return ptr;

            ptr = try_allocate_from_cache(aligned_size);
            if (ptr)
                return ptr;

            return allocate_new_block(size, aligned_size, "");
        }
    }

    void* try_allocate_from_prefill(size_t size, size_t aligned_size, bool is_prefill_context) {
        if (!is_prefill_context || !is_prefill_mode_ || !prefill_buffer_) {
            return nullptr;
        }

        if (prefill_buffer_used_ + aligned_size <= prefill_buffer_size_) {
            void* ptr = static_cast<char*>(prefill_buffer_) + prefill_buffer_used_;
            prefill_buffer_used_ += aligned_size;
            active_allocations_[ptr] = aligned_size;
            return ptr;
        }

        if (prefill_buffer_size_ < prefill_max_size_) {
            size_t new_size = std::min(prefill_buffer_size_ * 3 / 2, prefill_max_size_);
            new_size = std::max(new_size, prefill_buffer_used_ + aligned_size);
            new_size = std::min(new_size, prefill_max_size_);

            void* new_buffer = nullptr;
            cudaError_t err = cudaMalloc(&new_buffer, new_size);
            if (err != cudaSuccess) {
                return nullptr;
            }

            if (prefill_buffer_used_ > 0) {
                cudaError_t copy_err =
                    cudaMemcpy(new_buffer, prefill_buffer_, prefill_buffer_used_, cudaMemcpyDeviceToDevice);
                if (copy_err != cudaSuccess) {
                    cudaFree(new_buffer);
                    return nullptr;
                }
            }

            cudaFree(prefill_buffer_);
            prefill_buffer_ = new_buffer;
            prefill_buffer_size_ = new_size;

            void* ptr = static_cast<char*>(prefill_buffer_) + prefill_buffer_used_;
            prefill_buffer_used_ += aligned_size;
            active_allocations_[ptr] = aligned_size;
            return ptr;
        }
        return nullptr;
    }

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

    void* allocate_new_block(size_t size, size_t aligned_size, const std::string& tag = "") {
        size_t free_memory = 0, total_memory = 0;
        cudaError_t mem_err = cudaMemGetInfo(&free_memory, &total_memory);
        if (mem_err == cudaSuccess && aligned_size > free_memory * 0.8) {
            trim_threshold_internal(2);
            cudaMemGetInfo(&free_memory, &total_memory);
            if (aligned_size > free_memory * 0.8) {
                trim_internal();
            }
        }

        void* ptr = nullptr;
        cudaError_t err = cudaMalloc(&ptr, aligned_size);
        if (err != cudaSuccess) {
            return nullptr;
        }

        active_allocations_[ptr] = aligned_size;

        if (!tag.empty()) {
            tagged_memory_[tag] = ptr;
            memory_tags_[ptr] = tag;
            tagged_memory_sizes_[tag] = aligned_size;
            tagged_memory_active_[tag] = true;
        }
        return ptr;
    }

    void free(void* ptr) {
        if (!ptr)
            return;

        std::lock_guard<std::mutex> lock(mutex_);
        if (is_shutting_down_) {
            free_during_shutdown(ptr);
            return;
        }

        auto it_active = active_allocations_.find(ptr);
        if (it_active == active_allocations_.end()) {
            return;
        }
        size_t aligned_size = it_active->second;

        if (is_from_prefill_buffer(ptr)) {
            auto tag_it_prefill = memory_tags_.find(ptr);
            if (tag_it_prefill != memory_tags_.end()) {
                tagged_memory_active_[tag_it_prefill->second] = false;
            }
            free_from_prefill_buffer(ptr, aligned_size);
            active_allocations_.erase(it_active);
            return;
        }

        auto tag_it = memory_tags_.find(ptr);
        if (tag_it != memory_tags_.end()) {
            std::string tag = tag_it->second;
            tagged_memory_active_[tag] = false;
            active_allocations_.erase(it_active);
            return;
        }

        if (should_cache_block(aligned_size)) {
            free_blocks_[aligned_size].push_back(ptr);
        } else {
            cudaError_t err = cudaFree(ptr);
            if (err != cudaSuccess) {
                handle_cuda_error(err, aligned_size);
            }
        }
        active_allocations_.erase(it_active);
        perform_periodic_cleanup();
    }

    void free_during_shutdown(void* ptr) {
        auto it = active_allocations_.find(ptr);
        if (it != active_allocations_.end()) {
            active_allocations_.erase(it);
        }
    }

    bool is_from_prefill_buffer(void* ptr) {
        if (!is_prefill_mode_ || !prefill_buffer_)
            return false;
        char* prefill_start = static_cast<char*>(prefill_buffer_);
        char* prefill_end = prefill_start + prefill_buffer_size_;
        char* ptr_char = static_cast<char*>(ptr);
        return (ptr_char >= prefill_start && ptr_char < prefill_end);
    }

    void free_from_prefill_buffer(void* ptr, size_t aligned_size) {
    }

    bool should_cache_block(size_t aligned_size) {
        size_t max_blocks = 8;
        if (aligned_size >= 1024 * 1024) {
            max_blocks = 2;
        } else if (aligned_size >= 65536) {
            max_blocks = 4;
        }
        return free_blocks_[aligned_size].size() < max_blocks;
    }

    void handle_cuda_error(cudaError_t err, size_t aligned_size) {
        if (err == cudaErrorCudartUnloading) {
            is_shutting_down_ = true;
        } else {
            // cudaFree failed
        }
    }

    void* allocate_tagged(const std::string& tag, size_t size, bool is_prefill_request = false) {
        if (tag.empty()) {
            return allocate(size, is_prefill_request);
        }
        if (size == 0)
            return nullptr;

        std::lock_guard<std::mutex> lock(mutex_);
        if (is_shutting_down_)
            return nullptr;

        size_t aligned_size = (size + 255) & ~255;

        auto it_tag_mem = tagged_memory_.find(tag);
        if (it_tag_mem != tagged_memory_.end()) {
            void* ptr = it_tag_mem->second;
            size_t old_block_size = tagged_memory_sizes_[tag];
            bool is_block_active = tagged_memory_active_[tag];

            if (!is_block_active) {
                if (old_block_size >= aligned_size) {
                    tagged_memory_active_[tag] = true;
                    active_allocations_[ptr] = old_block_size;
                    return ptr;
                } else {
                    cudaError_t free_err = cudaFree(ptr);
                    if (free_err != cudaSuccess && free_err != cudaErrorInvalidDevicePointer &&
                        free_err != cudaErrorCudartUnloading) {
                        // Failed to free undersized inactive tagged block
                    }
                }
            } else {
                if (old_block_size >= aligned_size) {
                    return ptr;
                } else {
                    cudaError_t free_err = cudaFree(ptr);
                    if (free_err != cudaSuccess && free_err != cudaErrorInvalidDevicePointer &&
                        free_err != cudaErrorCudartUnloading) {
                        // Failed to free undersized active tagged block
                    }
                    active_allocations_.erase(ptr);
                }
            }

            tagged_memory_.erase(it_tag_mem);
            memory_tags_.erase(ptr);
            tagged_memory_sizes_.erase(tag);
            tagged_memory_active_.erase(tag);
        }

        // 修复：Tagged memory始终从独立的cudaMalloc分配，不使用prefill buffer
        // 这样可以避免与prefill阶段的临时内存产生冲突
        // Tagged memory通常是需要跨prefill阶段保持稳定的固定内存
        return allocate_new_block(size, aligned_size, tag);
    }

    void* get_tagged_memory(const std::string& tag) {
        if (tag.empty()) {
            return nullptr;
        }

        std::lock_guard<std::mutex> lock(mutex_);

        auto it = tagged_memory_.find(tag);
        if (it == tagged_memory_.end()) {
            return nullptr;
        }

        void* ptr = it->second;

        // 修改：无条件返回tagged memory，不检查active状态
        // 这样可以确保tagged memory始终可用于写入
        return ptr;
    }

    bool has_tag(const std::string& tag) {
        if (tag.empty()) {
            return false;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        return tagged_memory_.count(tag) > 0;
    }

    void perform_periodic_cleanup() {
        size_t total_cached_blocks = 0;
        for (const auto& [size_key, blocks] : free_blocks_) {
            total_cached_blocks += blocks.size();
        }

        if (total_cached_blocks > 100) {
            trim_threshold_internal(4);
        }
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

        if (is_shutting_down_) {
            return false;
        }

        prefill_max_size_ = max_size;

        if (is_prefill_mode_) {
            if (prefill_buffer_ != nullptr && prefill_buffer_size_ >= initial_size) {
                prefill_buffer_used_ = 0;
                return true;
            } else {
                if (prefill_buffer_ != nullptr) {
                    cudaFree(prefill_buffer_);
                    prefill_buffer_ = nullptr;
                }
                prefill_buffer_size_ = 0;
                prefill_buffer_used_ = 0;
                is_prefill_mode_ = false;
            }
        }

        size_t adjusted_initial_size = adjust_prefill_size(initial_size);

        cudaError_t err = cudaMalloc(&prefill_buffer_, adjusted_initial_size);
        if (err != cudaSuccess) {
            std::cerr << "CudaMemoryPool: Error - Failed to allocate prefill buffer: "
                      << (adjusted_initial_size / (1024.0 * 1024.0)) << " MB, Error: " << cudaGetErrorString(err)
                      << std::endl;
            prefill_buffer_ = nullptr;
            prefill_buffer_size_ = 0;
            prefill_buffer_used_ = 0;
            is_prefill_mode_ = false;
            return false;
        }

        prefill_buffer_size_ = adjusted_initial_size;
        prefill_buffer_used_ = 0;
        is_prefill_mode_ = true;

        std::cerr << "CudaMemoryPool: Prefill mode enabled. Initial: " << (prefill_buffer_size_ / (1024.0 * 1024.0))
                  << " MB, Max: " << (prefill_max_size_ / (1024.0 * 1024.0)) << " MB" << std::endl;
        return true;
    }

    size_t adjust_prefill_size(size_t requested_size) {
        size_t free_memory = 0, total_memory = 0;
        cudaError_t mem_err = cudaMemGetInfo(&free_memory, &total_memory);
        if (mem_err == cudaSuccess) {
            if (requested_size > free_memory * 0.8) {
                size_t adjusted_size = static_cast<size_t>(free_memory * 0.7);
                std::cerr << "CudaMemoryPool: Warning - Requested prefill size ("
                          << (requested_size / (1024.0 * 1024.0))
                          << "MB) is too large. Adjusted to: " << (adjusted_size / (1024.0 * 1024.0))
                          << " MB (70% of free " << (free_memory / (1024.0 * 1024.0)) << "MB)" << std::endl;
                return adjusted_size > (1024 * 1024) ? adjusted_size : (1024 * 1024);
            }
        } else {
            // Failed to get GPU memory info
        }
        return requested_size;
    }

    void disable_prefill_mode() {
        std::lock_guard<std::mutex> lock(mutex_);
        disable_prefill_mode_internal();
    }

    void reset_prefill_buffer() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!is_prefill_mode_ || prefill_buffer_ == nullptr)
            return;
        prefill_buffer_used_ = 0;
    }

    std::pair<size_t, size_t> count_active_prefill_allocations() const {
        size_t count = 0;
        size_t bytes = 0;
        if (!prefill_buffer_)
            return {count, bytes};

        char* buffer_start = static_cast<char*>(prefill_buffer_);
        char* buffer_end = buffer_start + prefill_buffer_size_;

        for (const auto& [ptr, alloc_size] : active_allocations_) {
            char* ptr_char = static_cast<char*>(ptr);
            if (ptr_char >= buffer_start && ptr_char < buffer_end) {
                count++;
                bytes += alloc_size;
            }
        }
        return {count, bytes};
    }

    void set_prefill_phase(bool is_prefill) {
        std::lock_guard<std::mutex> lock(mutex_);
        is_prefill_phase_ = is_prefill;

        if (is_prefill && !is_prefill_mode_ && !is_shutting_down_) {
            enable_prefill_mode_internal();
        }

        if (!is_prefill && is_prefill_mode_) {
            bool has_active_tagged_memory = false;
            for (const auto& [tag, is_active] : tagged_memory_active_) {
                if (is_active) {
                    has_active_tagged_memory = true;
                    break;
                }
            }

            if (!has_active_tagged_memory) {
                prefill_buffer_used_ = 0;
            }
        }
    }

    void prepare_for_shutdown() {
        std::lock_guard<std::mutex> lock(mutex_);
        is_shutting_down_ = true;
        free_blocks_.clear();
        std::cerr << "CudaMemoryPool: Prepared for safe shutdown. Further CUDA "
                     "operations will be restricted."
                  << std::endl;
    }

    bool is_shutting_down() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return is_shutting_down_;
    }

    struct PoolStats {
        size_t total_cached_blocks = 0;
        size_t total_cached_bytes = 0;
        size_t active_allocations_count = 0;
        size_t active_bytes = 0;
        size_t size_categories_in_cache = 0;
        size_t tagged_allocations_count = 0;
        size_t tagged_active_count = 0;
        size_t tagged_total_bytes = 0;
        size_t prefill_buffer_size_bytes = 0;
        size_t prefill_buffer_used_bytes = 0;
    };

    PoolStats getStats() const {
        std::lock_guard<std::mutex> lock(mutex_);
        PoolStats stats;

        stats.size_categories_in_cache = free_blocks_.size();
        for (const auto& [size_val, blocks] : free_blocks_) {
            stats.total_cached_blocks += blocks.size();
            stats.total_cached_bytes += size_val * blocks.size();
        }

        stats.active_allocations_count = active_allocations_.size();
        stats.active_bytes = std::accumulate(active_allocations_.begin(), active_allocations_.end(), 0ULL,
                                             [](size_t sum, const auto& pair) { return sum + pair.second; });

        stats.tagged_allocations_count = tagged_memory_.size();
        for (const auto& [tag, active_status] : tagged_memory_active_) {
            if (active_status) {
                stats.tagged_active_count++;
            }
        }
        for (const auto& [tag, size_val] : tagged_memory_sizes_) {
            stats.tagged_total_bytes += size_val;
        }

        stats.prefill_buffer_size_bytes = prefill_buffer_size_;
        stats.prefill_buffer_used_bytes = prefill_buffer_used_;
        return stats;
    }

   private:
    void enable_prefill_mode_internal(size_t initial_size = 48 * 1024 * 1024, size_t max_size = 512 * 1024 * 1024) {
        if (is_shutting_down_)
            return;
        prefill_max_size_ = max_size;

        if (is_prefill_mode_) {
            if (prefill_buffer_ != nullptr && prefill_buffer_size_ >= initial_size) {
                prefill_buffer_used_ = 0;
                return;
            } else {
                if (prefill_buffer_ != nullptr) {
                    cudaFree(prefill_buffer_);
                    prefill_buffer_ = nullptr;
                }
                prefill_buffer_size_ = 0;
                prefill_buffer_used_ = 0;
                is_prefill_mode_ = false;
            }
        }

        size_t adjusted_initial_size = adjust_prefill_size_internal(initial_size);
        cudaError_t err = cudaMalloc(&prefill_buffer_, adjusted_initial_size);
        if (err != cudaSuccess) {
            std::cerr << "CudaMemoryPool (Internal): Error - Failed to allocate "
                         "prefill buffer: "
                      << (adjusted_initial_size / (1024.0 * 1024.0)) << " MB, Error: " << cudaGetErrorString(err)
                      << std::endl;
            prefill_buffer_ = nullptr;
            prefill_buffer_size_ = 0;
            prefill_buffer_used_ = 0;
            is_prefill_mode_ = false;
            return;
        }
        prefill_buffer_size_ = adjusted_initial_size;
        prefill_buffer_used_ = 0;
        is_prefill_mode_ = true;
        std::cerr << "CudaMemoryPool (Internal): Prefill mode enabled. Initial: "
                  << (prefill_buffer_size_ / (1024.0 * 1024.0))
                  << " MB, Max: " << (prefill_max_size_ / (1024.0 * 1024.0)) << " MB" << std::endl;
    }

    size_t adjust_prefill_size_internal(size_t requested_size) {
        size_t free_memory = 0, total_memory = 0;
        cudaError_t mem_err = cudaMemGetInfo(&free_memory, &total_memory);
        if (mem_err == cudaSuccess) {
            if (requested_size > free_memory * 0.8) {
                size_t adjusted_size = static_cast<size_t>(free_memory * 0.7);
                std::cerr << "CudaMemoryPool (Internal): Warning - Requested prefill size ("
                          << (requested_size / (1024.0 * 1024.0))
                          << "MB) is too large. Adjusted to: " << (adjusted_size / (1024.0 * 1024.0)) << " MB."
                          << std::endl;
                return adjusted_size > (1024 * 1024) ? adjusted_size : (1024 * 1024);
            }
        }
        return requested_size;
    }

    void disable_prefill_mode_internal() {
        if (!is_prefill_mode_)
            return;

        if (prefill_buffer_ != nullptr) {
            cudaError_t driver_status = cudaFree(0);
            if (driver_status == cudaSuccess || driver_status == cudaErrorInvalidDevicePointer) {
                cudaError_t err = cudaFree(prefill_buffer_);
                if (err != cudaSuccess) {
                    std::cerr << "CudaMemoryPool: Warning - Failed to free prefill "
                                 "buffer during disable: "
                              << cudaGetErrorString(err) << std::endl;
                    if (err == cudaErrorCudartUnloading) {
                        is_shutting_down_ = true;
                    }
                }
            } else {
                is_shutting_down_ = true;
            }
            prefill_buffer_ = nullptr;
        }

        prefill_buffer_size_ = 0;
        prefill_buffer_used_ = 0;
        is_prefill_mode_ = false;
        std::cerr << "CudaMemoryPool: Prefill mode disabled." << std::endl;
    }

    void trim_internal(size_t size = 0) {
        if (size == 0) {
            for (auto& [block_size, blocks] : free_blocks_) {
                for (void* ptr : blocks) {
                    cudaError_t err = cudaFree(ptr);
                    if (err != cudaSuccess && err != cudaErrorCudartUnloading && err != cudaErrorInvalidDevicePointer) {
                        // Failed to free block
                    }
                }
            }
            free_blocks_.clear();
        } else {
            size_t aligned_size = (size + 255) & ~255;
            auto it = free_blocks_.find(aligned_size);
            if (it != free_blocks_.end()) {
                for (void* ptr : it->second) {
                    cudaError_t err = cudaFree(ptr);
                    if (err != cudaSuccess && err != cudaErrorCudartUnloading && err != cudaErrorInvalidDevicePointer) {
                        // Failed to free block
                    }
                }
                free_blocks_.erase(it);
            }
        }
    }

    void trim_threshold_internal(size_t max_blocks_per_size) {
        for (auto it = free_blocks_.begin(); it != free_blocks_.end();) {
            std::vector<void*>& blocks = it->second;
            while (blocks.size() > max_blocks_per_size) {
                void* ptr = blocks.back();
                cudaError_t err = cudaFree(ptr);
                if (err != cudaSuccess && err != cudaErrorCudartUnloading && err != cudaErrorInvalidDevicePointer) {
                    // Failed to free block
                }
                blocks.pop_back();
            }

            if (blocks.empty()) {
                it = free_blocks_.erase(it);
            } else {
                ++it;
            }
        }
    }

    std::map<std::string, void*> tagged_memory_;
    std::map<void*, std::string> memory_tags_;
    std::map<std::string, size_t> tagged_memory_sizes_;
    std::map<std::string, bool> tagged_memory_active_;

    std::unordered_map<size_t, std::vector<void*>> free_blocks_;
    std::unordered_map<void*, size_t> active_allocations_;

    bool is_prefill_mode_;
    void* prefill_buffer_;
    size_t prefill_buffer_size_;
    size_t prefill_buffer_used_;
    size_t prefill_max_size_;
    bool is_prefill_phase_;

    bool is_shutting_down_;

    mutable std::mutex mutex_;
};

class GlobalCudaMemoryPool {
   public:
    static CudaMemoryPool& instance();

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

    static void prepare_for_shutdown() {
        if (pool_instance_ptr != nullptr) {
            pool_instance_ptr->prepare_for_shutdown();
        } else {
            // Shutdown called before instance creation
        }
    }

    static bool is_shutting_down() {
        if (pool_instance_ptr != nullptr) {
            return pool_instance_ptr->is_shutting_down();
        }
        return false;
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
