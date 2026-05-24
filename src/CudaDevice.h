#pragma once

#include <string>
#include <vector>

class CudaDevice {
private:
    int deviceIndex;
    int picBusId;
public:
    CudaDevice(int index);
    int getDeviceIndex() const { return deviceIndex; }
    int getPicBusId() const { return picBusId; }

    std::string getName() const;
    std::string getFullName() const;

    static std::vector<CudaDevice> getAllDevices();
};
