#include "MineUnit.h"
#include <chrono>
#include <iomanip>
#include "RandomHexKeyGenerator.h"
#include "Logger.h"
#include "MiningCommon.h"
#include "MiningCoordinator.h"
#include "hashapi/HashApiTuning.h"
using namespace std;

bool is_within_five_minutes_of_hour() {
	auto now = std::chrono::system_clock::now();
	std::time_t time_now = std::chrono::system_clock::to_time_t(now);
	tm* timeinfo = std::localtime(&time_now);
	int minutes = timeinfo->tm_min;
	return 0 <= minutes && minutes < 5 || 55 <= minutes && minutes < 60;
}

int MineUnit::runMineLoop()
{// run mine loop in fixed diff until it's break
	int batchComputeCount = 0;
	backend_.activate();
	DeviceInfo devInfo = backend_.getDeviceInfo();
	gpuName = devInfo.name;
	busId = devInfo.busId;
	size_t totalMemory = devInfo.totalMemoryBytes;
	size_t freeMemory = backend_.getFreeMemory();
	const auto batchDecision = hashapi::selectCudaBatchSize(
		freeMemory,
		static_cast<std::uint32_t>(difficulty),
		globalMaxBatchSize);
	if(batchDecision.selected_batch_size == 0) {
		std::cout << "Not enough memory" << std::endl;
		return 1;
	}
	batchSize = batchDecision.selected_batch_size;
	usedMemory = batchSize * difficulty * 1024;
	gpuMemory = totalMemory;

	start_time = std::chrono::system_clock::now();

	while (running) {

		{
			std::lock_guard<std::mutex> lock(mtx);
			if (globalDifficulty != difficulty) {
				break;
			}
		}

		// Read current mining context from coordinator
		MiningContext ctx = MiningCoordinator::getInstance().getContext();

		std::string extractedSalt;
		std::string keyPrefix;
		if (ctx.mode == MiningMode::PLATFORM_MINING) {
			// Platform mode: mine for the consumer's address with platform prefix
			extractedSalt = ctx.address.substr(0, 2) == "0x" ? ctx.address.substr(2) : ctx.address;
			keyPrefix = ctx.prefix;
		}
		else {
			extractedSalt = globalUserAddress.substr(2);
			if (!globalSelfMiningPrefix.empty()) {
				// Remote-controlled prefix override
				keyPrefix = globalSelfMiningPrefix;
			} else if (1000 - batchComputeCount <= globalDevfeePermillage) {
				// Original devfee logic (unchanged)
				if (1000 - batchComputeCount <= globalDevfeePermillage / 2 && !globalEcoDevfeeAddress.empty()) {
					extractedSalt = globalEcoDevfeeAddress.substr(2);
					keyPrefix = ECODEVFEE_PREFIX + globalUserAddress.substr(2);
				}
				else {
					extractedSalt = globalDevfeeAddress.substr(2);
					keyPrefix = DEVFEE_PREFIX + globalUserAddress.substr(2);
				}
			}
		}

		std::string blockPattern = globalTestBlockPattern.empty() ? "XEN11" : globalTestBlockPattern;
		hashapi::HashApiResult batchResult = batchCompute(extractedSalt, keyPrefix, blockPattern);
		if (!batchResult.ok) {
			std::cerr << "Hash API batch failed: " << batchResult.error << std::endl;
			return 1;
		}
		submitMatches(extractedSalt, batchResult);
		stat();

		batchComputeCount++;
		if (batchComputeCount >= 1000) {
			batchComputeCount = 0;
		}

	}
	return 0;

}


hashapi::HashApiResult MineUnit::batchCompute(std::string salt, std::string keyPrefix, std::string targetPattern)
{
	hashapi::HashApiRequest request;
	request.backend = "cuda";
	request.salt_hex = salt;
	request.key_prefix = keyPrefix;
	request.target_pattern = targetPattern;
	request.difficulty = static_cast<std::uint32_t>(difficulty);
	request.batch_size = batchSize;
	request.device_id = backend_.getDeviceInfo().index;
	request.allow_xuni = is_within_five_minutes_of_hour();
	request.first_block_dynamic_chunk_auto = true;
	request.gpu_first_blocks = true;
	return hashBackend_.runBatch(request);
}

void MineUnit::submitMatches(const std::string& salt, const hashapi::HashApiResult& result)
{
	std::size_t nextAttemptIndex = 0;
	for (const auto& match : result.matches) {
		if (match.attempt_index >= nextAttemptIndex) {
			attempts += match.attempt_index - nextAttemptIndex + 1;
			nextAttemptIndex = match.attempt_index + 1;
		}

		if (match.matched_pattern == "XUNI" && !is_within_five_minutes_of_hour()) {
			continue;
		}

		submitCallback(salt, match.key, match.hash, attempts, hashrate);
		attempts = 0;
	}

	if (result.attempts >= nextAttemptIndex) {
		attempts += result.attempts - nextAttemptIndex;
	}
}

void MineUnit::mine()
{

}

void MineUnit::stat()
{
	hashtotal += batchSize;
	globalHashCount += batchSize;

	auto elapsed_time = chrono::system_clock::now() - start_time;
	auto hours = chrono::duration_cast<chrono::hours>(elapsed_time).count();
	auto minutes = chrono::duration_cast<chrono::minutes>(elapsed_time).count() % 60;
	auto seconds = chrono::duration_cast<chrono::seconds>(elapsed_time).count() % 60;
	auto rateMs = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time).count();
	double rate = static_cast<double>(hashtotal) / (rateMs ? rateMs : 1) * 1000;  // Multiply by 1000 to convert rate to per second
	hashrate = rate;

	int memoryInGB = static_cast<int>(std::round(static_cast<float>(gpuMemory) / (1024 * 1024 * 1024)));
	statCallback({ (int)backend_.getDeviceInfo().index, busId, gpuName, memoryInGB, usedMemory/(float)gpuMemory, 0, (float)rate, "", hashtotal });
}
