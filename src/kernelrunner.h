#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

class KernelRunner
{
private:
    uint32_t type, version;
    uint32_t passes, lanes, segmentBlocks;
    uint32_t allocatedSegmentBlocks;
    std::size_t batchSize;

    cudaEvent_t start, end, copyStart, copyEnd, firstBlockStart, firstBlockEnd, kernelStart, kernelEnd;
    cudaStream_t stream;
    void* memory;
    void* refs;
    void* deviceKeys;
    void* deviceSalt;
    std::size_t deviceKeysCapacity;
    std::size_t deviceSaltCapacity;
    bool deviceFirstBlocksReady;
    std::size_t deviceFirstBlockKeyLength;
    std::uint32_t deviceFirstBlockSaltLength;
    std::uint32_t deviceFirstBlockOutputLength;
    std::uint32_t deviceFirstBlockMemoryCost;
    std::uint32_t deviceFirstBlockTimeCost;
    std::uint32_t deviceFirstBlockVersion;
    std::uint32_t deviceFirstBlockType;
    std::uint32_t deviceFirstBlockLanes;
    bool lastUsedDeviceFirstBlocks;

    std::unique_ptr<uint8_t[]> blocksIn;
    std::unique_ptr<uint8_t[]> blocksOut;

    void copyInputBlocks();
    void copyOutputBlocks();

    void runDeviceFirstBlockKernel();
    void runKernelOneshot();

public:

    std::size_t getBatchSize() const { return batchSize; }

    KernelRunner(uint32_t type, uint32_t version,
        uint32_t passes, uint32_t lanes,
        uint32_t segmentBlocks, std::size_t batchSize);
    ~KernelRunner();

    void init(std::size_t batchSize);
    bool canReuse(uint32_t type, uint32_t version,
        uint32_t passes, uint32_t lanes,
        uint32_t segmentBlocks, std::size_t batchSize) const;
    void reconfigure(uint32_t type, uint32_t version,
        uint32_t passes, uint32_t lanes,
        uint32_t segmentBlocks, std::size_t batchSize);

    void* getInputMemory(std::size_t jobId) const;
    const void* getOutputMemory(std::size_t jobId) const;
    bool prepareInputBlocksOnDevice(const std::vector<std::string>& passwords,
                                    const std::vector<std::uint8_t>& saltBytes,
                                    std::uint32_t outputLength,
                                    std::uint32_t memoryCost,
                                    std::uint32_t timeCost,
                                    std::uint32_t version,
                                    std::uint32_t type,
                                    std::uint32_t lanes);

    void run();
    float finish();
    float getLastHostToDeviceMs() const;
    float getLastGpuFirstBlockMs() const;
    float getLastDeviceToHostMs() const;
};
