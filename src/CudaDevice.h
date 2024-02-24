#pragma once

#include <string>
#include <vector>

class CudaDevice {
private:
    int deviceIndex;
    int picBusId;
public:
    CudaDevice(int index);
    int getDeviceIndex() { return deviceIndex; }
    int getPicBusId() { return picBusId; }

    std::string getName();
    std::string getFullName();

    static std::vector<CudaDevice> getAllDevices();
};