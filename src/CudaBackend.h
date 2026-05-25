#pragma once

#include "ComputeBackend.h"
#include "CudaDevice.h"
#include "kernelrunner.h"
#include <memory>

class CudaBackend : public ComputeBackend {
public:
	explicit CudaBackend(int deviceIndex);

	DeviceInfo getDeviceInfo() const override;
	size_t getFreeMemory() const override;
	void activate() override;
	void init(size_t batchSize, uint32_t type, uint32_t version,
	          uint32_t passes, uint32_t lanes,
	          uint32_t segmentBlocks) override;
	void* getInputMemory(size_t jobId) const override;
	const void* getOutputMemory(size_t jobId) const override;
	bool prepareInputBlocksOnDevice(const std::vector<std::string>& passwords,
	                                const std::vector<std::uint8_t>& saltBytes,
	                                std::uint32_t outputLength,
	                                std::uint32_t memoryCost,
	                                std::uint32_t timeCost,
	                                std::uint32_t version,
	                                std::uint32_t type,
	                                std::uint32_t lanes) override;
	void run() override;
	float finish() override;
	float getLastHostToDeviceMs() const override;
	float getLastGpuFirstBlockMs() const override;
	float getLastDeviceToHostMs() const override;

	static std::vector<std::unique_ptr<ComputeBackend>> enumerate();

private:
	int deviceIndex_;
	DeviceInfo info_;
	std::unique_ptr<KernelRunner> runner_;
};
