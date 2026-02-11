#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>

extern std::mutex mtx_submit;
extern std::condition_variable cv;
extern std::queue<std::function<void()>> taskQueue;

void workerThread();
