#include "StatReporter.h"

#include <iostream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <nvml.h>
#include "HttpClient.h"

extern std::string globalCustomName;

nlohmann::json vectorToJson(const std::string& machineId,
                            const std::string& accountAddress,
                            const std::vector<std::pair<int, gpuInfo>>& data)
{
    nlohmann::json j;
    nlohmann::json gpuArray = nlohmann::json::array();

    for (const auto &item : data)
    {
        nlohmann::json jItem;
        std::ostringstream os;

        jItem["index"] = item.first;
        jItem["name"] = item.second.name;
        jItem["memory"] = item.second.memory;

        os << std::fixed << std::setprecision(2) << item.second.usingMemory * 100;
        jItem["usingMemory"] = os.str();
        jItem["temperature"] = item.second.temperature;

        os.str("");
        os.clear();
        os << std::fixed << std::setprecision(2) << item.second.hashrate;
        jItem["hashrate"] = os.str();
        jItem["power"] = item.second.power;
        jItem["hashCount"] = item.second.hashCount;
        gpuArray.push_back(jItem);
    }

    j["machineId"] = machineId;
    j["accountAddress"] = accountAddress;
    j["gpuInfos"] = gpuArray;

    return j;
}

void uploadGpuInfos()
{
    while (running)
    {
        auto now = std::chrono::steady_clock::now();
        std::map<int, std::pair<gpuInfo, std::chrono::steady_clock::time_point>> gpuinfos;
        {
            std::lock_guard<std::mutex> lock(globalGpuInfosMutex);
            gpuinfos = globalGpuInfos;
        }
        std::vector<std::pair<int, gpuInfo>> gpuInfos;
        for (const auto &kv : gpuinfos)
        {
            auto duration = std::chrono::duration_cast<std::chrono::minutes>(now - kv.second.second);
            if (duration.count() <= 2)
            {
                gpuInfos.push_back({kv.first, kv.second.first});
            }
        }
        if (gpuInfos.size() == 0)
        {
            std::this_thread::sleep_for(std::chrono::minutes(5));
            continue;
        }
        std::string infoJson = vectorToJson(machineId, globalUserAddress, gpuInfos).dump(-1);
        std::this_thread::sleep_for(std::chrono::minutes(5));
    }
}

std::string getGpuStatsJson() {
    nlohmann::json result;
    nlohmann::json gpuArray = nlohmann::json::array();
    float totalHashrate = 0.0;

    std::lock_guard<std::mutex> guard(globalGpuInfosMutex);

    auto now = std::chrono::system_clock::now();
    auto uptime = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();

    for (const auto& gpuInfoPair : globalGpuInfos) {
        const auto& gpuInfo = gpuInfoPair.second.first;
        nlohmann::json gpuJson;
        gpuJson["index"] = gpuInfo.index;
        gpuJson["hashrate"] = gpuInfo.hashrate;
        gpuJson["busId"] = gpuInfo.busId;
        totalHashrate += gpuInfo.hashrate;
        gpuArray.push_back(gpuJson);
    }

    result["totalHashrate"] = totalHashrate;
    result["gpus"] = gpuArray;
    result["uptime"] = uptime;
    result["acceptedBlocks"] = globalNormalBlockCount.load() + globalSuperBlockCount.load();
    result["rejectedBlocks"] = globalFailedBlockCount.load();

    return result.dump();
}

nlohmann::json getStatData() {
    nlohmann::json result;
    nlohmann::json gpuArray = nlohmann::json::array();
    float totalHashrate = 0.0;
    size_t totalHashCount = 0;

    std::lock_guard<std::mutex> guard(globalGpuInfosMutex);

    auto now = std::chrono::system_clock::now();
    auto uptime = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
    nvmlReturn_t nvmlResult;
    nvmlDevice_t nvmlDevice;
    nvmlUtilization_t nvmlUtilization;
    nvmlMemory_t nvmlMemory;
    unsigned int totalPower;
    nvmlResult = nvmlInit();
    for (const auto& gpuInfoPair : globalGpuInfos) {
        const auto& gpuInfo = gpuInfoPair.second.first;
        nlohmann::json gpuJson;
        gpuJson["index"] = gpuInfo.index;
        gpuJson["name"] = gpuInfo.name;
        std::ostringstream stream_hashRate;
        stream_hashRate << std::fixed << std::setprecision(2) << gpuInfo.hashrate;
        gpuJson["hashrate"] = stream_hashRate.str();
        gpuJson["memory"] = gpuInfo.memory;
        unsigned int power = -1;
        if(nvmlResult == NVML_SUCCESS) {
            nvmlReturn_t nvmlResult_ = nvmlDeviceGetHandleByIndex(gpuInfo.index, &nvmlDevice);
            if (nvmlResult_ == NVML_SUCCESS) {
                nvmlResult_ = nvmlDeviceGetPowerUsage(nvmlDevice, &power);
                nvmlResult_ = nvmlDeviceGetUtilizationRates(nvmlDevice, &nvmlUtilization);
            }
        }
        gpuJson["power"] = power;
        totalPower += power == -1 ? 0 : power;
        gpuJson["utiliz"] = nvmlUtilization.gpu;
        std::ostringstream stream_usingMemory;
        stream_usingMemory << std::fixed << std::setprecision(1) << gpuInfo.usingMemory * 100;
        gpuJson["usingMemory"] = stream_usingMemory.str();
        gpuJson["hashCount"] = gpuInfo.hashCount;
        totalHashrate += gpuInfo.hashrate;
        totalHashCount += gpuInfo.hashCount;
        gpuArray.push_back(gpuJson);
    }
    if(nvmlResult == NVML_SUCCESS) {
        nvmlShutdown();
    }

    result["machineId"] = machineId;
    result["minerAddr"] = globalUserAddress;
    std::ostringstream stream_totalHashrate;
    stream_totalHashrate << std::fixed << std::setprecision(2) << totalHashrate;
    result["totalHashrate"] = stream_totalHashrate.str();
    result["totalHashCount"] = totalHashCount;
    result["totalPower"] = totalPower;
    int difficulty = 40404;
    {
        std::lock_guard<std::mutex> lock(mtx);
        difficulty = globalDifficulty;
    }
    result["difficulty"] = difficulty;
    result["gpus"] = gpuArray;
    result["uptime"] = uptime;
    result["acceptedBlocks"] = globalNormalBlockCount.load() + globalSuperBlockCount.load();
    result["normalBlocks"] = globalNormalBlockCount.load();
    result["superBlocks"] = globalSuperBlockCount.load();
    result["rejectedBlocks"] = globalFailedBlockCount.load();
    result["version"] = "2.0.0";
    if (!globalCustomName.empty()) {
        result["customName"] = globalCustomName;
    }
    return result;
}

void UploadDataPeriodically(int uploadPeriod) {
    HttpClient client;
    std::string url = "https://woodyminer.com/api/stat/upload";
    long timeout = 3000;
    int failureCount = 0;
    int originalUploadPeriod = uploadPeriod;
    std::this_thread::sleep_for(std::chrono::seconds(10));

    while (running) {
        auto data = getStatData();
        auto response = client.HttpPost(url, data, timeout);
        if (response.GetStatusCode() == 201) {
            failureCount = 0;
            uploadPeriod = originalUploadPeriod;
        } else {
            failureCount++;
        }
        if (failureCount >= 10) {
            uploadPeriod *= 2;
            if(uploadPeriod > 600) {
                uploadPeriod = 600;
            }
        }
        std::this_thread::sleep_for(std::chrono::seconds(uploadPeriod));
    }
}
