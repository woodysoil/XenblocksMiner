#include "CudaDevice.h"
#include <stdexcept>
#include <cuda_runtime.h>
#include <sstream>
#include <cmath>

#include "CudaException.h"

std::string CudaDevice::getName() {
    cudaDeviceProp prop;
    CudaException::check(cudaGetDeviceProperties(&prop, deviceIndex));
    return std::string(prop.name);
}

std::string CudaDevice::getFullName() {
    cudaDeviceProp prop;
    CudaException::check(cudaGetDeviceProperties(&prop, deviceIndex));

    std::string name = prop.name;

    int memoryInGB = static_cast<int>(std::round(static_cast<float>(prop.totalGlobalMem) / (1024 * 1024 * 1024)));

    std::ostringstream fullNameStream;
    fullNameStream << name << " | " << memoryInGB << " GB";

    return fullNameStream.str();
}

std::vector<CudaDevice> CudaDevice::getAllDevices() {
    int count;
    CudaException::check(cudaGetDeviceCount(&count));

    std::vector<CudaDevice> devices;
    devices.reserve(count);
    for (int i = 0; i < count; i++) {
        devices.emplace_back(i);
    }
    return devices;
}
