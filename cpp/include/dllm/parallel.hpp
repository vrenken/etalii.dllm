// A small persistent thread pool with fixed-partition semantics.
//
// Kernels split their work into tasks that each own a disjoint set of output elements and compute them in the one
// documented order. Which thread runs a task, and how many threads there are, therefore cannot change a single bit
// of the result: the thread count is a speed setting, never a numeric one. Tasks are handed out through an atomic
// counter; nothing is ever reduced across tasks. See docs/kernels.md ("Threads and SIMD").
#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#include "fpenv.hpp"

#if defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#define DLLM_HAS_FORK 1
#endif

namespace dllm {

class ThreadPool {
public:
    // The process-wide pool. It is created on first use and never destroyed: joining threads from static
    // destructors can deadlock while a Windows DLL unloads, and blocked workers do not keep a process alive.
    static ThreadPool& global() {
        static ThreadPool* pool = new ThreadPool(default_threads());
        return *pool;
    }

    // DLLM_THREADS if set to a positive number, else the hardware concurrency.
    static std::size_t default_threads() {
        if (const char* value = std::getenv("DLLM_THREADS")) {
            const long n = std::strtol(value, nullptr, 10);
            if (n > 0) {
                return static_cast<std::size_t>(n);
            }
        }
        const unsigned hardware = std::thread::hardware_concurrency();
        return hardware == 0 ? 1 : hardware;
    }

    std::size_t threads() const { return threads_.load(); }

    // Changes the number of threads (the caller counts as one); 0 restores the default.
    void set_threads(std::size_t n) {
        std::lock_guard<std::mutex> run(run_mutex_);
        stop_workers();
        threads_ = n == 0 ? default_threads() : n;
    }

    // Runs fn(task) for every task in [0, tasks) and returns when all are done. When the pool is already busy (a
    // kernel called from another thread) the tasks simply run on the calling thread, which gives the same bits.
    void run(std::size_t tasks, const std::function<void(std::size_t)>& fn) {
        std::unique_lock<std::mutex> run(run_mutex_, std::try_to_lock);
        const std::size_t threads = threads_.load();
        if (tasks <= 1 || threads <= 1 || !run.owns_lock()) {
            for (std::size_t t = 0; t < tasks; ++t) {
                fn(t);
            }
            return;
        }
        start_workers(threads - 1);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            job_ = &fn;
            tasks_ = tasks;
            next_ = 0;
            pending_ = workers_.size();
            ++generation_;
        }
        wake_.notify_all();
        work();
        std::unique_lock<std::mutex> lock(mutex_);
        done_.wait(lock, [this] { return pending_ == 0; });
        job_ = nullptr;
    }

private:
    explicit ThreadPool(std::size_t threads) : threads_(threads) {}

    void work() {
        for (std::size_t t = next_.fetch_add(1); t < tasks_; t = next_.fetch_add(1)) {
            (*job_)(t);
        }
    }

    void start_workers(std::size_t count) {
#ifdef DLLM_HAS_FORK
        // Threads do not survive fork(): a child process starts its own workers (the parent's are abandoned).
        if (owner_ != getpid()) {
            for (auto& worker : workers_) {
                worker.detach();
            }
            workers_.clear();
            owner_ = getpid();
        }
#endif
        if (workers_.size() == count) {
            return;
        }
        stop_workers();
        stopping_ = false;
        for (std::size_t i = 0; i < count; ++i) {
            workers_.emplace_back([this, seen = generation_]() mutable {
                enter_canonical_fp_environment();  // a new thread may inherit a non-default state
                for (;;) {
                    {
                        std::unique_lock<std::mutex> lock(mutex_);
                        wake_.wait(lock, [&] { return stopping_ || generation_ != seen; });
                        if (stopping_) {
                            return;
                        }
                        seen = generation_;
                    }
                    work();
                    std::lock_guard<std::mutex> lock(mutex_);
                    if (--pending_ == 0) {
                        done_.notify_one();
                    }
                }
            });
        }
    }

    void stop_workers() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        wake_.notify_all();
        for (auto& worker : workers_) {
            worker.join();
        }
        workers_.clear();
    }

    std::atomic<std::size_t> threads_;
    std::mutex run_mutex_;  // one parallel job at a time
    std::mutex mutex_;
    std::condition_variable wake_;
    std::condition_variable done_;
    std::vector<std::thread> workers_;
    const std::function<void(std::size_t)>* job_ = nullptr;
    std::size_t tasks_ = 0;
    std::atomic<std::size_t> next_{0};
    std::size_t pending_ = 0;
    std::size_t generation_ = 0;
    bool stopping_ = false;
#ifdef DLLM_HAS_FORK
    pid_t owner_ = getpid();
#endif
};

// Runs fn(task) for task in [0, tasks) on the global pool.
inline void parallel_for(std::size_t tasks, const std::function<void(std::size_t)>& fn) {
    ThreadPool::global().run(tasks, fn);
}

}  // namespace dllm
