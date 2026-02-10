#include "MqttClient.h"
#include "MiningCommon.h"

#include <iostream>
#include <chrono>
#include <thread>

MqttClient::MqttClient(const std::string& broker_uri, const std::string& worker_id)
	: broker_uri_(broker_uri), worker_id_(worker_id)
{
	client_id_ = "xenminer_" + worker_id_;
	client_ = std::make_unique<mqtt::async_client>(broker_uri_, client_id_);
	client_->set_callback(*this);
}

MqttClient::~MqttClient()
{
	disconnect();
}

bool MqttClient::connect()
{
	if (connected_) return true;

	try {
		auto conn_opts = mqtt::connect_options_builder()
			.clean_session(true)
			.automatic_reconnect(
				std::chrono::milliseconds(INITIAL_RECONNECT_DELAY_MS),
				std::chrono::milliseconds(MAX_RECONNECT_DELAY_MS))
			.keep_alive_interval(std::chrono::seconds(KEEPALIVE_INTERVAL_SEC))
			.connect_timeout(std::chrono::seconds(CONNECT_TIMEOUT_SEC))
			.finalize();

		// Set Last Will and Testament (LWT) so broker knows if we disconnect ungracefully
		nlohmann::json lwt_payload = {
			{"worker_id", worker_id_},
			{"status", "offline"},
			{"timestamp", std::chrono::duration_cast<std::chrono::seconds>(
				std::chrono::system_clock::now().time_since_epoch()).count()}
		};
		std::string lwt_topic = buildTopic(TOPIC_STATUS);
		mqtt::message_ptr lwt_msg = mqtt::make_message(lwt_topic, lwt_payload.dump(), QOS, true);
		conn_opts.set_will_message(lwt_msg);

		std::cout << "MQTT: Connecting to " << broker_uri_ << "..." << std::endl;
		auto tok = client_->connect(conn_opts);
		tok->wait();
		connected_ = true;
		std::cout << GREEN << "MQTT: Connected successfully" << RESET << std::endl;
		return true;
	} catch (const mqtt::exception& e) {
		std::cerr << RED << "MQTT: Connection failed: " << e.what() << RESET << std::endl;
		return false;
	}
}

void MqttClient::disconnect()
{
	if (!connected_) return;

	try {
		// Publish offline status before disconnecting
		nlohmann::json offline = {
			{"worker_id", worker_id_},
			{"status", "offline"},
			{"timestamp", std::chrono::duration_cast<std::chrono::seconds>(
				std::chrono::system_clock::now().time_since_epoch()).count()}
		};
		publish(buildTopic(TOPIC_STATUS), offline);

		connected_ = false;
		client_->disconnect()->wait();
		std::cout << "MQTT: Disconnected" << std::endl;
	} catch (const mqtt::exception& e) {
		std::cerr << "MQTT: Disconnect error: " << e.what() << std::endl;
	}
}

bool MqttClient::isConnected() const
{
	return connected_ && client_ && client_->is_connected();
}

bool MqttClient::publish(const std::string& topic, const nlohmann::json& payload)
{
	return publish(topic, payload.dump());
}

bool MqttClient::publish(const std::string& topic, const std::string& payload)
{
	if (!isConnected()) return false;

	try {
		auto msg = mqtt::make_message(topic, payload, QOS, false);
		client_->publish(msg);
		return true;
	} catch (const mqtt::exception& e) {
		std::cerr << "MQTT: Publish failed on " << topic << ": " << e.what() << std::endl;
		return false;
	}
}

bool MqttClient::subscribe(const std::string& topic)
{
	if (!isConnected()) return false;

	try {
		client_->subscribe(topic, QOS)->wait();
		{
			std::lock_guard<std::mutex> lock(topics_mutex_);
			subscribed_topics_.push_back(topic);
		}
		return true;
	} catch (const mqtt::exception& e) {
		std::cerr << "MQTT: Subscribe failed on " << topic << ": " << e.what() << std::endl;
		return false;
	}
}

void MqttClient::setMessageCallback(MqttMessageCallback callback)
{
	std::lock_guard<std::mutex> lock(callback_mutex_);
	message_callback_ = std::move(callback);
}

std::string MqttClient::buildTopic(const std::string& suffix) const
{
	return "xenminer/" + worker_id_ + "/" + suffix;
}

// --- mqtt::callback overrides ---

void MqttClient::connected(const std::string& cause)
{
	connected_ = true;
	std::cout << GREEN << "MQTT: Connected" << RESET;
	if (!cause.empty()) {
		std::cout << " (reconnect: " << cause << ")";
	}
	std::cout << std::endl;
	resubscribeAll();
}

void MqttClient::connection_lost(const std::string& cause)
{
	connected_ = false;
	std::cout << YELLOW << "MQTT: Connection lost";
	if (!cause.empty()) {
		std::cout << " - " << cause;
	}
	std::cout << RESET << std::endl;
	// Automatic reconnect is handled by Paho's built-in reconnect
}

void MqttClient::message_arrived(mqtt::const_message_ptr msg)
{
	std::lock_guard<std::mutex> lock(callback_mutex_);
	if (message_callback_) {
		message_callback_(msg->get_topic(), msg->to_string());
	}
}

// --- mqtt::iaction_listener overrides ---

void MqttClient::on_failure(const mqtt::token& tok)
{
	std::cerr << "MQTT: Action failed, token: " << tok.get_message_id() << std::endl;
}

void MqttClient::on_success(const mqtt::token& tok)
{
	// No-op for async success
}

void MqttClient::resubscribeAll()
{
	std::lock_guard<std::mutex> lock(topics_mutex_);
	for (const auto& topic : subscribed_topics_) {
		try {
			client_->subscribe(topic, QOS);
		} catch (const mqtt::exception& e) {
			std::cerr << "MQTT: Re-subscribe failed on " << topic << ": " << e.what() << std::endl;
		}
	}
}
