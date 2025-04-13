#pragma once

#include <atomic>
#include <condition_variable>
#include <future>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <vector>

// ----- 通用任务类 -----

class Task {
 public:
  virtual ~Task() = default;
  virtual void execute() = 0;
};

class OpTask : public Task {
 public:
  // 构造函数在此处内联实现，防止重复定义
  explicit OpTask(std::function<void()> op) : op_(std::move(op)) {}
  void execute() override { op_(); }
 private:
  std::function<void()> op_;
};

// ----- 线程池类 -----

class ThreadPool {
 public:
  explicit ThreadPool(size_t numThreads)
      : stopFlag(false), taskCount(0) {
    for (size_t i = 0; i < numThreads; ++i) {
      workers.emplace_back([this]() {
        while (true) {
          std::shared_ptr<Task> task;
          {
            std::unique_lock<std::mutex> lock(queueMutex);
            condition.wait(lock, [this]() {
              return stopFlag.load() || !taskQueue.empty();
            });
            if (stopFlag.load() && taskQueue.empty()) {
              return;
            }
            task = taskQueue.front();
            taskQueue.pop();
            ++taskCount;
          }
          task->execute();
          {
            std::unique_lock<std::mutex> lock(completionMutex);
            --taskCount;
            if (taskQueue.empty() && taskCount.load() == 0)
              completionCondition.notify_all();
          }
        }
      });
    }
  }

  ~ThreadPool() {
    stopThreadPool();
    for (std::thread &worker : workers) {
      if (worker.joinable())
        worker.join();
    }
  }

  // 禁止拷贝构造和拷贝赋值
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  // 无返回值的任务入队
  void enqueueTask(std::shared_ptr<Task> task) {
    {
      std::unique_lock<std::mutex> lock(queueMutex);
      taskQueue.push(task);
    }
    condition.notify_one();
  }

  // 模板版本任务入队，返回 future
  template <typename F, typename... Args>
  auto enqueue(F&& f, Args&&... args)
      -> std::future<typename std::result_of<F(Args...)>::type> {
    using return_type = typename std::result_of<F(Args...)>::type;
    auto task_ptr = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...));
    std::future<return_type> res = task_ptr->get_future();
    {
      std::unique_lock<std::mutex> lock(queueMutex);
      if (stopFlag.load())
        throw std::runtime_error("enqueue on stopped ThreadPool");
      taskQueue.push(std::make_shared<OpTask>([task_ptr]() { (*task_ptr)(); }));
    }
    condition.notify_one();
    return res;
  }

  // 停止线程池
  void stopThreadPool() {
    {
      std::unique_lock<std::mutex> lock(queueMutex);
      stopFlag.store(true);
    }
    condition.notify_all();
  }

  // 等待所有任务完成
  void waitForAllTasks() {
    std::unique_lock<std::mutex> lock(completionMutex);
    completionCondition.wait(lock, [this]() {
      return taskQueue.empty() && taskCount.load() == 0;
    });
  }

 private:
  std::vector<std::thread> workers;               // 工作线程集合
  std::queue<std::shared_ptr<Task>> taskQueue;      // 任务队列
  std::mutex queueMutex;                          // 保护任务队列的锁
  std::condition_variable condition;              // 用于线程等待的条件变量
  std::atomic<bool> stopFlag;                       // 停止标志

  std::atomic<size_t> taskCount;                    // 记录正在执行或等待的任务数
  std::mutex completionMutex;                       // 用于等待任务完成的锁
  std::condition_variable completionCondition;      // 用于通知所有任务完成的条件变量
};
