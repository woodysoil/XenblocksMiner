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

	virtual void run() = 0;
	virtual float finish() = 0;
};

// Enumerate all available compute devices for the compiled backend.
std::vector<std::unique_ptr<ComputeBackend>> enumerateBackends();
