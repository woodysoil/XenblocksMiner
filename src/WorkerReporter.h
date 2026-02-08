#pragma once

#include <string>
#include <memory>
#include <vector>
#include "MqttClient.h"
#include "MiningCommon.h"

// WorkerReporter handles registration, heartbeat, and status reporting
// to the platform via MQTT.
class WorkerReporter
{
public:
	explicit WorkerReporter(std::shared_ptr<MqttClient> mqtt);

	// Send worker registration with GPU capabilities
	bool sendRegistration(const std::string& eth_address,
						  const std::vector<gpuInfo>& gpus);

	// Send periodic heartbeat with current stats
	bool sendHeartbeat(float total_hashrate,
					   int active_gpus,
					   int accepted_blocks);

	// Report mining status change
	bool sendStatusUpdate(const std::string& state,
						  const std::string& lease_id = "",
						  const std::string& detail = "");

	// Report a found block to the platform
	bool sendBlockFound(const std::string& lease_id,
						const std::string& hash,
						const std::string& key,
						const std::string& account,
						size_t attempts,
						float hashrate);

private:
	int64_t currentTimestamp() const;

	std::shared_ptr<MqttClient> mqtt_;
};
