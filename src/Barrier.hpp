#pragma once
#include <mutex>
#include <condition_variable>

struct Barrier {
    std::mutex mtx;
    std::condition_variable cv;
    int count;
    int waiting = 0;
    int generation = 0;

    explicit Barrier(int n) : count(n) {}

    void arrive_and_wait() {
        std::unique_lock<std::mutex> lock(mtx);
        int gen = generation;
        waiting++;
        if (waiting == count) {
            waiting = 0;
            generation++;
            cv.notify_all();
        } else {
            cv.wait(lock, [&] { return gen != generation; });
        }
    }
};
