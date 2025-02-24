#include "thread_pool.hpp"


OpTask::OpTask(std::function<void()> op) : op(std::move(op)) {}
void OpTask::execute() {
    op();
}

ThreadPool::ThreadPool(size_t numThreads) : stop(false) {
    for (size_t i = 0; i < numThreads; i++) {
        workers.emplace_back([this]() {
            while (true) {
                std::shared_ptr<Task> task;
                {
                   
                    std::unique_lock<std::mutex> lock(this->queueMutex);
                    this->condition.wait(lock, [this]() {
                        return this->stop || !this->taskQueue.empty();
                    });

               
                    if (this->stop && this->taskQueue.empty()) {
                        return;
                    }

                    task = std::move(this->taskQueue.front());
                    this->taskQueue.pop();
                }

               
                task->execute();
            }
        });
    }
}

// 向任务队列添加任务
void ThreadPool::enqueueTask(std::shared_ptr<Task> task) {
    {
        std::unique_lock<std::mutex> lock(queueMutex);
        if (stop) {
            throw std::runtime_error("enqueue on stopped ThreadPool");
        }
        taskQueue.push(std::move(task));
    }
    condition.notify_one();  
}


void ThreadPool::stopThreadPool() {
    {
        std::unique_lock<std::mutex> lock(queueMutex);
        stop = true;
    }
    condition.notify_all();  
    for (auto& worker : workers) {
        worker.join();
    }
}

// ThreadPool 的析构函数
ThreadPool::~ThreadPool() {
    stopThreadPool();
}
