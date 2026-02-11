#include "BlockSubmitter.h"

#include "MiningCommon.h"

std::mutex mtx_submit;
std::condition_variable cv;
std::queue<std::function<void()>> taskQueue;

void workerThread() {
    while (running) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(mtx_submit);
            cv.wait(lock, []{ return !running || !taskQueue.empty(); });

            if (!running && taskQueue.empty()) {
                break;
            }

            task = std::move(taskQueue.front());
            taskQueue.pop();
        }

        task();
    }
}
