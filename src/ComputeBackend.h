#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct DeviceInfo {
	int index;
	int busId;
	std::string name;
	size_t totalMemoryBytes;
};

class ComputeBackend {
public:
	virtual ~ComputeBackend() = default;

	virtual DeviceInfo getDeviceInfo() const = 0;
	virtual size_t getFreeMemory() const = 0;

	// Activate device for current thread (e.g. cudaSetDevice)
	virtual void activate() = 0;

	// Allocate buffers for batch Argon2 hashing.
	// Can be called multiple times; previous allocations are released.
	virtual void init(size_t batchSize, uint32_t type, uint32_t version,
	                  uint32_t passes, uint32_t lanes,
	                  uint32_t segmentBlocks) = 0;

	virtual void* getInputMemory(size_t jobId) const = 0;
	virtual const void* getOutputMemory(size_t jobId) const = 0;

	virtual bool prepareInputBlocksOnDevice(const std::vector<std::string>& passwords,
	                                        const std::vector<std::uint8_t>& saltBytes,
	                                        std::uint32_t outputLength,
	                                        std::uint32_t memoryCost,
	                                        std::uint32_t timeCost,
	                                        std::uint32_t version,
	                                        std::uint32_t type,
	                                        std::uint32_t lanes)
	{
		(void)passwords;
		(void)saltBytes;
		(void)outputLength;
		(void)memoryCost;
		(void)timeCost;
		(void)version;
		(void)type;
		(void)lanes;
		return false;
	}

	virtual void run() = 0;
	virtual float finish() = 0;
	virtual float getLastHostToDeviceMs() const { return 0.0f; }
	virtual float getLastGpuFirstBlockMs() const { return 0.0f; }
	virtual float getLastDeviceToHostMs() const { return 0.0f; }
};

// Enumerate all available compute devices for the compiled backend.
std::vector<std::unique_ptr<ComputeBackend>> enumerateBackends();
