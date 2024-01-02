#pragma once
#include <cuda_runtime.h>
#include <memory>

class KernelRunner
{
private:
    uint32_t type, version;
    uint32_t passes, lanes, segmentBlocks;
    std::size_t batchSize;

    cudaEvent_t start, end, kernelStart, kernelEnd;
    cudaStream_t stream;
    void* memory;
    void* refs;

    std::unique_ptr<uint8_t[]> blocksIn;
    std::unique_ptr<uint8_t[]> blocksOut;

    void copyInputBlocks();
    void copyOutputBlocks();

    void runKernelOneshot();

public:

    std::size_t getBatchSize() const { return batchSize; }

    KernelRunner(uint32_t type, uint32_t version,
        uint32_t passes, uint32_t lanes,
        uint32_t segmentBlocks, std::size_t batchSize);
    ~KernelRunner();

    void init(std::size_t batchSize);

    void* getInputMemory(std::size_t jobId) const;
    const void* getOutputMemory(std::size_t jobId) const;

    void run();
    float finish();
};
