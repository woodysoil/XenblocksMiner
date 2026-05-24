#pragma once
#include <iostream>
#include <chrono>
#include <vector>
#include <string>

#include "MiningCommon.h"
#include "argon2-common.h"
#include "argon2params.h"
#include "ComputeBackend.h"
#include "hashapi/CudaHashBackend.h"
#include "hashapi/HashApiTypes.h"

class RandomHexKeyGenerator;

class MineUnit
{
private:
	ComputeBackend& backend_;
	hashapi::CudaHashBackend hashBackend_;
	std::size_t difficulty;
	std::size_t batchSize = 1;
	SubmitCallback submitCallback;
	StatCallback statCallback;
	std::chrono::system_clock::time_point start_time;
	size_t hashtotal = 0;
	std::size_t attempts = 0;
	float hashrate = 0;
	std::string gpuName;
	std::size_t gpuMemory = 0;
	std::size_t usedMemory = 0;
	int busId;

public:
	MineUnit(ComputeBackend& backend, std::size_t difficulty,
		SubmitCallback submitCallback, StatCallback statCallback)
		: backend_(backend), hashBackend_(backend), difficulty(difficulty),
		submitCallback(submitCallback), statCallback(statCallback)
	{
	}

	int runMineLoop();
	hashapi::HashApiResult batchCompute(std::string salt, std::string keyPrefix, std::string targetPattern);
private:
	void submitMatches(const std::string& salt, const hashapi::HashApiResult& result);
private:
	void mine();
	void stat();
};
