#pragma once

#include <string>
#include <vector>
#include <utility>
#include <nlohmann/json.hpp>
#include "MiningCommon.h"

nlohmann::json vectorToJson(const std::string& machineId,
                            const std::string& accountAddress,
                            const std::vector<std::pair<int, gpuInfo>>& data);
void uploadGpuInfos();
std::string getGpuStatsJson();
nlohmann::json getStatData();
void UploadDataPeriodically(int uploadPeriod);
