#pragma once
#include <iostream>
#include <chrono>
#include <vector>
#include <string>

#include "MiningCommon.h"
#include "argon2-common.h"
#include "argon2params.h"
#include "ComputeBackend.h"

class RandomHexKeyGenerator;

struct HashItem {
	std::string key;
	std::string hashed;
};

class MineUnit
{
private:
	ComputeBackend& backend_;
	std::size_t difficulty;
	std::size_t batchSize = 1;
	Argon2Params params;
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

	std::vector<std::string> passwordStorage;
public:
	MineUnit(ComputeBackend& backend, std::size_t difficulty,
		SubmitCallback submitCallback, StatCallback statCallback)
		: backend_(backend), difficulty(difficulty),
		submitCallback(submitCallback), statCallback(statCallback)
	{
	}

	int runMineLoop();
	std::vector<HashItem> batchCompute(RandomHexKeyGenerator& keyGenerator, std::string salt);
private:
	void setPassword(std::size_t index, std::string pwd);
	void getHash(std::size_t index, void* hash);
	std::string getPW(std::size_t index);
private:
	void mine();
	void stat();
};
