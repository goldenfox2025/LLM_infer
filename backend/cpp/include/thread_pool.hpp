#ifndef THREAD_POOL_HPP
#define THREAD_POOL_HPP

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <vector>


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
  void waitForAllTasks();  // 新增等待所有任务完成的方法

 private:
  std::vector<std::thread> workers;             // 工作线程
  std::queue<std::shared_ptr<Task>> taskQueue;  // 任务队列
  std::mutex queueMutex;                        // 任务队列锁
  std::condition_variable condition;            // 条件变量
  std::atomic<bool> stop;                       // 停止标志

  std::atomic<size_t> taskCount;                // 原子计数器，记录任务数量
  std::mutex completionMutex;                   // 完成条件变量的互斥锁
  std::condition_variable completionCondition;  // 完成条件变量
};

#endif  // THREAD_POOL_HPP
