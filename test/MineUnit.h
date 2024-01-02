#pragma once
#include <iostream>
#include <chrono>
#include <vector>
#include <string>

#include "MiningCommon.h"
#include "argon2-common.h"
#include "argon2params.h"
#include "kernelrunner.h"

class RandomHexKeyGenerator;

struct HashItem {
	std::string key;
	std::string hashed;
};

class MineUnit
{
private:
	std::size_t deviceIndex;
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

	std::vector<std::string> passwordStorage;
	KernelRunner kernelRunner;
public:
	MineUnit(std::size_t deviceIndex, std::size_t difficulty,
		SubmitCallback submitCallback, StatCallback statCallback)
		: deviceIndex(deviceIndex), difficulty(difficulty),
		submitCallback(submitCallback), statCallback(statCallback),
		kernelRunner(argon2::ARGON2_ID, argon2::ARGON2_VERSION_13, 1,
			1, Argon2Params(argon2::ARGON2_ID, argon2::ARGON2_VERSION_13, HASH_LENGTH, "abcdef", NULL, 0, NULL, 0,
				1, difficulty, 1).getSegmentBlocks(), batchSize)
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

