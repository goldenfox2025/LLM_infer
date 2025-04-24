#include "CudaMemoryPool.hpp"
#include <mutex>

// 定义静态成员变量
CudaMemoryPool* GlobalCudaMemoryPool::pool_instance_ptr = nullptr;
std::once_flag GlobalCudaMemoryPool::init_flag_;

// 获取全局单例实例
CudaMemoryPool& GlobalCudaMemoryPool::instance() {
    // 使用 std::call_once 确保 pool_instance_ptr 在多线程环境下只被初始化一次
    std::call_once(init_flag_, []() {
        // std::cerr << "正在初始化全局 CudaMemoryPool 单例..." << std::endl;
        pool_instance_ptr = new CudaMemoryPool();

        // 注册程序退出时的清理函数，确保在所有其他对象销毁后执行
        std::atexit([]() {
            // 在程序退出时执行清理
            std::cerr << "Executing final CUDA memory cleanup..." << std::endl;

            // 获取当前活跃分配的数量
            auto stats = pool_instance_ptr->getStats();
            if (stats.active_allocations > 0) {
                std::cerr << "Warning: " << stats.active_allocations
                        << " active CUDA memory allocations ("
                        << stats.active_bytes << " bytes) at program exit."
                        << std::endl;
            }

            // 释放所有缓存的内存块
            pool_instance_ptr->trim();

            // 注意：我们不释放活跃分配，因为它们可能仍然被使用
            // 操作系统会在程序退出时回收所有内存

            // 重要：我们故意不删除 pool_instance_ptr
            // 这确保了内存池实例的生命周期持续到程序进程结束，
            // 从而允许其他静态/全局对象在其析构函数中安全地调用 free()
            // delete pool_instance_ptr; // <<-- 不要这样做！
        });
    });

    // 返回创建好的实例的引用
    return *pool_instance_ptr;
}
