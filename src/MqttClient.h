#pragma once

#include <string>
#include <functional>
#include <mutex>
#include <atomic>
#include <memory>
#include <mqtt/async_client.h>
#include <nlohmann/json.hpp>

// Callback for incoming MQTT messages
using MqttMessageCallback = std::function<void(const std::string& topic, const std::string& payload)>;

class MqttClient : public virtual mqtt::callback, public virtual mqtt::iaction_listener
{
public:
	MqttClient(const std::string& broker_uri, const std::string& worker_id);
	~MqttClient();

	// Connection management
	bool connect();
	void disconnect();
	bool isConnected() const;

	// Publish with QoS 1
	bool publish(const std::string& topic, const nlohmann::json& payload);
	bool publish(const std::string& topic, const std::string& payload);

	// Subscribe to a topic with QoS 1
	bool subscribe(const std::string& topic);

	// Set callback for incoming messages
	void setMessageCallback(MqttMessageCallback callback);

	// Topic helpers - builds full topic path: xenminer/{worker_id}/{suffix}
	std::string buildTopic(const std::string& suffix) const;

	// Pre-defined topic suffixes
	static constexpr const char* TOPIC_REGISTER   = "register";
	static constexpr const char* TOPIC_HEARTBEAT  = "heartbeat";
	static constexpr const char* TOPIC_STATUS     = "status";
	static constexpr const char* TOPIC_BLOCK      = "block";
	static constexpr const char* TOPIC_TASK       = "task";
	static constexpr const char* TOPIC_CONTROL    = "control";

private:
	// mqtt::callback overrides
	void connected(const std::string& cause) override;
	void connection_lost(const std::string& cause) override;
	void message_arrived(mqtt::const_message_ptr msg) override;

	// mqtt::iaction_listener overrides
	void on_failure(const mqtt::token& tok) override;
	void on_success(const mqtt::token& tok) override;

	void resubscribeAll();

	std::string broker_uri_;
	std::string worker_id_;
	std::string client_id_;
	std::unique_ptr<mqtt::async_client> client_;

	MqttMessageCallback message_callback_;
	std::mutex callback_mutex_;

	std::vector<std::string> subscribed_topics_;
	std::mutex topics_mutex_;

	std::atomic<bool> connected_{false};

	static constexpr int QOS = 1;
	static constexpr int MAX_RECONNECT_DELAY_MS = 30000;
	static constexpr int INITIAL_RECONNECT_DELAY_MS = 1000;
	static constexpr int CONNECT_TIMEOUT_SEC = 10;
	static constexpr int KEEPALIVE_INTERVAL_SEC = 60;
};
