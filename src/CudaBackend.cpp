#include "CudaBackend.h"
#include "CudaException.h"
#include <cuda_runtime.h>
#include <cmath>

CudaBackend::CudaBackend(int deviceIndex)
	: deviceIndex_(deviceIndex)
{
	cudaDeviceProp prop;
	CudaException::check(cudaGetDeviceProperties(&prop, deviceIndex_));
	info_ = {
		deviceIndex_,
		prop.pciBusID,
		std::string(prop.name),
		prop.totalGlobalMem,
	};
}

DeviceInfo CudaBackend::getDeviceInfo() const
{
	return info_;
}

size_t CudaBackend::getFreeMemory() const
{
	size_t freeMem = 0, totalMem = 0;
	cudaMemGetInfo(&freeMem, &totalMem);
	return freeMem;
}

void CudaBackend::activate()
{
	CudaException::check(cudaSetDevice(deviceIndex_));
}

void CudaBackend::init(size_t batchSize, uint32_t type, uint32_t version,
                       uint32_t passes, uint32_t lanes,
                       uint32_t segmentBlocks)
{
	runner_ = std::make_unique<KernelRunner>(type, version, passes, lanes,
	                                        segmentBlocks, batchSize);
	runner_->init(batchSize);
}

void* CudaBackend::getInputMemory(size_t jobId) const
{
	return runner_->getInputMemory(jobId);
}

const void* CudaBackend::getOutputMemory(size_t jobId) const
{
	return runner_->getOutputMemory(jobId);
}

void CudaBackend::run()
{
	runner_->run();
}

float CudaBackend::finish()
{
	return runner_->finish();
}

std::vector<std::unique_ptr<ComputeBackend>> CudaBackend::enumerate()
{
	auto devices = CudaDevice::getAllDevices();
	std::vector<std::unique_ptr<ComputeBackend>> backends;
	backends.reserve(devices.size());
	for (const auto& dev : devices) {
		backends.push_back(std::make_unique<CudaBackend>(dev.getDeviceIndex()));
	}
	return backends;
}

std::vector<std::unique_ptr<ComputeBackend>> enumerateBackends()
{
	return CudaBackend::enumerate();
}
