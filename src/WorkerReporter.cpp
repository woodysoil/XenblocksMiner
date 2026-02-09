#include "WorkerReporter.h"

#include <chrono>
#include <sstream>
#include <iomanip>
#include <iostream>

WorkerReporter::WorkerReporter(std::shared_ptr<MqttClient> mqtt)
	: mqtt_(std::move(mqtt))
{
}

bool WorkerReporter::sendRegistration(const std::string& eth_address,
									  const std::vector<gpuInfo>& gpus)
{
	nlohmann::json gpu_array = nlohmann::json::array();
	int total_memory = 0;
	for (const auto& gpu : gpus) {
		gpu_array.push_back({
			{"index", gpu.index},
			{"name", gpu.name},
			{"memory_gb", gpu.memory},
			{"bus_id", gpu.busId}
		});
		total_memory += gpu.memory;
	}

	nlohmann::json payload = {
		{"worker_id", machineId},
		{"eth_address", eth_address},
		{"gpu_count", static_cast<int>(gpus.size())},
		{"total_memory_gb", total_memory},
		{"gpus", gpu_array},
		{"version", "2.0.0"},
		{"timestamp", currentTimestamp()}
	};

	std::string topic = mqtt_->buildTopic(MqttClient::TOPIC_REGISTER);
	bool ok = mqtt_->publish(topic, payload);
	if (ok) {
		std::cout << "MQTT: Registration sent (" << gpus.size() << " GPUs, "
				  << total_memory << " GB)" << std::endl;
	}
	return ok;
}

bool WorkerReporter::sendHeartbeat(float total_hashrate,
								   int active_gpus,
								   int accepted_blocks)
{
	nlohmann::json payload = {
		{"worker_id", machineId},
		{"hashrate", total_hashrate},
		{"active_gpus", active_gpus},
		{"accepted_blocks", accepted_blocks},
		{"difficulty", globalDifficulty.load()},
		{"uptime_sec", std::chrono::duration_cast<std::chrono::seconds>(
			std::chrono::system_clock::now() - start_time).count()},
		{"timestamp", currentTimestamp()}
	};

	return mqtt_->publish(mqtt_->buildTopic(MqttClient::TOPIC_HEARTBEAT), payload);
}

bool WorkerReporter::sendStatusUpdate(const std::string& state,
									  const std::string& lease_id,
									  const std::string& detail)
{
	nlohmann::json payload = {
		{"worker_id", machineId},
		{"state", state},
		{"timestamp", currentTimestamp()}
	};

	if (!lease_id.empty()) {
		payload["lease_id"] = lease_id;
	}
	if (!detail.empty()) {
		payload["detail"] = detail;
	}

	return mqtt_->publish(mqtt_->buildTopic(MqttClient::TOPIC_STATUS), payload);
}

bool WorkerReporter::sendBlockFound(const std::string& lease_id,
									const std::string& hash,
									const std::string& key,
									const std::string& account,
									size_t attempts,
									float hashrate)
{
	std::ostringstream hr_stream;
	hr_stream << std::fixed << std::setprecision(2) << hashrate;

	nlohmann::json payload = {
		{"worker_id", machineId},
		{"lease_id", lease_id},
		{"hash", hash},
		{"key", key},
		{"account", account},
		{"attempts", attempts},
		{"hashrate", hr_stream.str()},
		{"timestamp", currentTimestamp()}
	};

	std::string topic = mqtt_->buildTopic(MqttClient::TOPIC_BLOCK);
	bool ok = mqtt_->publish(topic, payload);
	if (ok) {
		if (lease_id.empty())
			std::cout << GREEN << "MQTT: Self-mined block reported" << RESET << std::endl;
		else
			std::cout << GREEN << "MQTT: Block reported for lease " << lease_id << RESET << std::endl;
	}
	return ok;
}

int64_t WorkerReporter::currentTimestamp() const
{
	return std::chrono::duration_cast<std::chrono::seconds>(
		std::chrono::system_clock::now().time_since_epoch()).count();
}
