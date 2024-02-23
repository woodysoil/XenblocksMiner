#include "MineUnit.h"
#include <chrono>
#include <regex>
#include <iomanip>
#include "RandomHexKeyGenerator.h"
#include "Logger.h"
#include "CudaDevice.h"
#include "MiningCommon.h"
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
	cudaSetDevice(deviceIndex);
	gpuName = CudaDevice(deviceIndex).getName();

	size_t freeMemory, totalMemory;
	cudaMemGetInfo(&freeMemory, &totalMemory);
	batchSize = freeMemory / 1.001 / difficulty / 1024;
	usedMemory = batchSize * difficulty * 1024;
	gpuMemory = totalMemory;

	start_time = std::chrono::system_clock::now();
	RandomHexKeyGenerator keyGenerator("", HASH_LENGTH);
	kernelRunner.init(batchSize);
	while (running) {
		
		{
			std::lock_guard<std::mutex> lock(mtx);
			if (globalDifficulty != difficulty) {
				break;
			}
		}

		std::string extractedSalt = globalUserAddress.substr(2);
		if (1000 - batchComputeCount <= globalDevfeePermillage) {
			if (1000 - batchComputeCount <= globalDevfeePermillage / 2 && !globalEcoDevfeeAddress.empty()) {
				extractedSalt = globalEcoDevfeeAddress.substr(2);
				keyGenerator.setPrefix(ECODEVFEE_PREFIX + globalUserAddress.substr(2));
			}
			else {
				extractedSalt = globalDevfeeAddress.substr(2);
				keyGenerator.setPrefix(DEVFEE_PREFIX + globalUserAddress.substr(2));
			}
		}
		else {
			keyGenerator.setPrefix("");
		}

		std::vector<HashItem> batchItems = batchCompute(keyGenerator, extractedSalt);

		std::regex pattern(R"(XUNI\d)");
		for (const auto& item : batchItems) {
			attempts++;
			if (item.hashed.find("XEN11") != std::string::npos) {
				// std::cout << "XEN11 found Hash " << item.hashed << std::endl;
				submitCallback(extractedSalt, item.key, item.hashed, attempts, hashrate);
				attempts = 0;
			}

			if (std::regex_search(item.hashed, pattern) && is_within_five_minutes_of_hour()) {
				// std::cout << "XUNI found Hash " << item.hashed << std::endl;
				submitCallback(extractedSalt, item.key, item.hashed, attempts, hashrate);
				attempts = 0;
			}

		}
		stat();

		batchComputeCount++;
		if (batchComputeCount >= 1000) {
			batchComputeCount = 0;
		}

	}
	return 0;

}


static const std::string base64_chars =
"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
"abcdefghijklmnopqrstuvwxyz"
"0123456789+/";

std::string base64_encode(unsigned char const* bytes_to_encode, unsigned int in_len) {
	std::string ret;
	int i = 0;
	int j = 0;
	unsigned char char_array_3[3];
	unsigned char char_array_4[4];

	while (in_len--) {
		char_array_3[i++] = *(bytes_to_encode++);
		if (i == 3) {
			char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
			char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
			char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
			char_array_4[3] = char_array_3[2] & 0x3f;

			for (i = 0; (i < 4); i++)
				ret += base64_chars[char_array_4[i]];
			i = 0;
		}
	}

	if (i) {
		for (j = i; j < 3; j++)
			char_array_3[j] = '\0';

		char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
		char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
		char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
		char_array_4[3] = char_array_3[2] & 0x3f;

		for (j = 0; (j < i + 1); j++)
			ret += base64_chars[char_array_4[j]];
	}

	return ret;
}

std::vector<HashItem> MineUnit::batchCompute(RandomHexKeyGenerator& keyGenerator, std::string salt)
{
	Argon2Params paramsTmp(argon2::ARGON2_ID, argon2::ARGON2_VERSION_13, HASH_LENGTH, salt, nullptr, 0, nullptr, 0, 1, difficulty, 1);
	this->params = paramsTmp;
	for (std::size_t i = 0; i < batchSize; i++) {
		setPassword(i, keyGenerator.nextRandomKey());
	}

	kernelRunner.run();
	kernelRunner.finish();

	std::vector<HashItem> hashItems;

	for (std::size_t i = 0; i < batchSize; i++) {
		uint8_t buffer[HASH_LENGTH];
		getHash(i, buffer);
		std::string decodedString = base64_encode(buffer, HASH_LENGTH);
		std::string key = getPW(i);

		hashItems.push_back({ key, decodedString });
	}

	return hashItems;
}

void MineUnit::setPassword(std::size_t index, std::string pwd)
{
	params.fillFirstBlocks(kernelRunner.getInputMemory(index), pwd.c_str(), pwd.size());

	if (passwordStorage.size() <= index) {
		passwordStorage.resize(index + 1);
	}

	passwordStorage[index] = pwd;
}

void MineUnit::getHash(std::size_t index, void* hash)
{
	params.finalize(hash, kernelRunner.getOutputMemory(index));
}

std::string MineUnit::getPW(std::size_t index)
{
	if (index < passwordStorage.size()) {
		return passwordStorage[index];
	}
	return {};
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
	statCallback({ (int)deviceIndex, gpuName, memoryInGB, usedMemory/(float)gpuMemory, 0, (float)rate, "", hashtotal });
}
