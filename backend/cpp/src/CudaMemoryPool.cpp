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

            // 检查CUDA驱动程序是否仍然可用
            cudaError_t driver_status = cudaFree(0);
            bool driver_available = (driver_status == cudaSuccess ||
                                    driver_status == cudaErrorInvalidDevicePointer);

            if (!driver_available) {
                std::cerr << "CUDA driver is shutting down or unavailable, skipping detailed cleanup" << std::endl;
                // 首先将内存池设置为关闭状态，防止新的分配和释放操作
                GlobalCudaMemoryPool::prepare_for_shutdown();
                return;
            }

            // 首先将内存池设置为关闭状态，防止新的分配和释放操作
            GlobalCudaMemoryPool::prepare_for_shutdown();

            // 获取当前活跃分配的数量
            auto stats = pool_instance_ptr->getStats();
            if (stats.active_allocations > 0) {
                std::cerr << "Warning: " << stats.active_allocations
                        << " active CUDA memory allocations ("
                        << stats.active_bytes << " bytes) at program exit."
                        << std::endl;

                // 打印prefill模式的统计信息
                if (stats.prefill_buffer_size > 0) {
                    std::cerr << "Prefill buffer: " << (stats.prefill_buffer_size / (1024 * 1024))
                            << " MB total, " << (stats.prefill_buffer_used / (1024 * 1024))
                            << " MB used" << std::endl;
                }
            }

            // 注意：我们不释放活跃分配，因为它们可能仍然被使用
            // 操作系统会在程序退出时回收所有内存

            // 我们故意不删除 pool_instance_ptr
            // 这确保了内存池实例的生命周期持续到程序进程结束，
            // 从而允许其他静态/全局对象在其析构函数中安全地调用 free()
            // delete pool_instance_ptr; // <<-- 不要这样做！
        });
    });

    // 返回创建好的实例的引用
    return *pool_instance_ptr;
}
