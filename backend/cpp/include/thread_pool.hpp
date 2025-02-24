#ifndef THREAD_POOL_HPP
#define THREAD_POOL_HPP

#include <functional>
#include <memory>
#include <queue>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <stdexcept>

// 假设 Tensor 是一个模板类，在其他地方有定义
template<typename T>
class Tensor;

// 通用任务类，可以接受任意算子函数
class Task {
public:
    virtual ~Task() = default;
    virtual void execute() = 0;
};

class OpTask : public Task {
public:
    explicit OpTask(std::function<void()> op);
    void execute() override;

private:
    std::function<void()> op;
};

class ThreadPool {
public:
    explicit ThreadPool(size_t numThreads);
    ~ThreadPool();
    
    // 禁止拷贝构造和拷贝赋值
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    void enqueueTask(std::shared_ptr<Task> task);
    void stopThreadPool();

private:
    std::vector<std::thread> workers;                // 工作线程
    std::queue<std::shared_ptr<Task>> taskQueue;     // 任务队列
    std::mutex queueMutex;                           // 任务队列锁
    std::condition_variable condition;               // 条件变量
    std::atomic<bool> stop;                          // 停止标志
};

#endif // THREAD_POOL_HPP
