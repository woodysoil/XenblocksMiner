#include "MiningCommon.h"

std::atomic<bool> running = true;
std::atomic<int> globalDifficulty = 1727;
std::mutex mtx;
std::mutex coutmtx;

std::string globalUserAddress = "0x123456789";
std::string globalDevfeeAddress = "0x24691E54aFafe2416a8252097C9Ca67557271475";
std::string globalEcoDevfeeAddress = "";
std::atomic<int> globalDevfeePermillage = 1; // per 1000
std::string machineId = "00000";

std::map<int, std::pair<gpuInfo, std::chrono::steady_clock::time_point>> globalGpuInfos;
std::mutex globalGpuInfosMutex;

std::atomic<int> globalNormalBlockCount = 0;
std::atomic<int> globalSuperBlockCount = 0;
std::atomic<int> globalXuniBlockCount = 0;
std::atomic<int> globalFailedBlockCount = 0;

std::atomic<long> globalHashCount = 0;
std::chrono::system_clock::time_point start_time = std::chrono::system_clock::now();
