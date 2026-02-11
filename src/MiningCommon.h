#pragma once

#include <chrono>
#include <mutex>
#include <atomic>
#include <functional>
#include <string>
#include <map>
#include <chrono>

constexpr std::size_t HASH_LENGTH = 64;
constexpr std::size_t MAX_SUBMIT_RETRIES = 5;

const std::string CONFIG_FILENAME = "config.txt";
const std::string DEVFEE_PREFIX = "FFFFFFFF";
const std::string ECODEVFEE_PREFIX = "EEEEEEEE";

// --- Hashpower Marketplace types ---

constexpr std::size_t PLATFORM_PREFIX_LENGTH = 16;

enum class MiningMode {
	SELF_MINING,
	PLATFORM_MINING
};

struct MiningContext {
	MiningMode mode = MiningMode::SELF_MINING;
	std::string address;       // target mining address (user's own or consumer's)
	std::string prefix;        // hex prefix for key generation (16 chars for platform)
	std::string consumer_id;   // platform consumer identifier
	std::string lease_id;      // platform lease identifier
};

extern std::string globalUserAddress;
extern std::string globalDevfeeAddress;
extern std::string globalEcoDevfeeAddress;
extern std::atomic<int> globalDevfeePermillage; // per 1000
extern std::string machineId;

extern std::atomic<int> globalDifficulty;
extern std::mutex mtx;
extern std::atomic<bool> running;
extern std::mutex coutmtx;

extern std::atomic<int> globalNormalBlockCount;
extern std::atomic<int> globalSuperBlockCount;
extern std::atomic<int> globalXuniBlockCount;
extern std::atomic<int> globalFailedBlockCount;

extern std::chrono::system_clock::time_point start_time;
extern std::atomic<long> globalHashCount;

extern std::string globalRpcLink;
extern std::string globalTestBlockPattern;
extern std::string globalSelfMiningPrefix;
extern std::size_t globalMaxBatchSize;

struct gpuInfo
{
	int index;
	int busId;
	std::string name;
	int memory;
	float usingMemory;
	int temperature;
	float hashrate;
	std::string power;
	size_t hashCount;
};
extern std::map<int, std::pair<gpuInfo, std::chrono::steady_clock::time_point>> globalGpuInfos;
extern std::mutex globalGpuInfosMutex;

using SubmitCallback = std::function<void(const std::string& hexsalt, const std::string& key, const std::string& hashed_pure, const size_t attempts, const float hashrate)>;
using StatCallback = std::function<void(const gpuInfo gpuinfo)>;

struct MinerConfig {
	std::string userAddress;
	std::string devfeeAddress;
	std::string ecoDevfeeAddress;
	std::atomic<int> devfeePermillage{0};
	std::string rpcLink;
	std::string testBlockPattern;
	std::string selfMiningPrefix;
	std::size_t maxBatchSize = 0;
	std::string customName;
	bool platformMode = false;
	std::string mqttBroker;
	std::string workerId;
};

const std::string RED = "\033[31m";
const std::string GREEN = "\033[32m";
const std::string YELLOW = "\033[33m";
const std::string RESET = "\033[0m";
