#pragma once

#include <string>
#include <vector>

class CudaDevice {
private:
    int deviceIndex;

public:
    CudaDevice(int index) : deviceIndex(index) {}
    int getDeviceIndex() { return deviceIndex; }

    std::string getName();
    std::string getFullName();

    static std::vector<CudaDevice> getAllDevices();
};