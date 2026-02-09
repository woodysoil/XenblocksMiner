#include "PlatformManager.h"

#include <iostream>
#include <chrono>

const char* platformStateToString(PlatformState state)
{
	switch (state) {
		case PlatformState::IDLE:      return "IDLE";
		case PlatformState::AVAILABLE: return "AVAILABLE";
		case PlatformState::LEASED:    return "LEASED";
		case PlatformState::MINING:    return "MINING";
		case PlatformState::COMPLETED: return "COMPLETED";
		case PlatformState::ERROR:     return "ERROR";
		default:                       return "UNKNOWN";
	}
}

PlatformManager::PlatformManager(const std::string& broker_uri,
								 const std::string& eth_address,
								 const std::vector<gpuInfo>& gpus)
	: mqtt_(std::make_shared<MqttClient>(broker_uri, machineId)),
	  reporter_(mqtt_),
	  eth_address_(eth_address),
	  gpus_(gpus)
{
}

PlatformManager::~PlatformManager()
{
	stop();
}

bool PlatformManager::start()
{
	if (running_) return true;

	// Connect to MQTT broker
	if (!mqtt_->connect()) {
		std::cerr << RED << "PlatformManager: Failed to connect to MQTT broker" << RESET << std::endl;
		transitionTo(PlatformState::ERROR);
		return false;
	}

	running_ = true;

	// Subscribe to incoming command topics
	mqtt_->subscribe(mqtt_->buildTopic(MqttClient::TOPIC_TASK));
	mqtt_->subscribe(mqtt_->buildTopic(MqttClient::TOPIC_CONTROL));

	// Set up message handler
	mqtt_->setMessageCallback([this](const std::string& topic, const std::string& payload) {
		onMessage(topic, payload);
	});

	// Send registration
	reporter_.sendRegistration(eth_address_, gpus_);

	// Transition to AVAILABLE (will be confirmed by register_ack)
	// For now, go AVAILABLE optimistically; handleRegisterAck confirms it
	transitionTo(PlatformState::AVAILABLE);

	// Start heartbeat thread
	heartbeat_thread_ = std::thread(&PlatformManager::heartbeatLoop, this);

	// Start lease watchdog thread
	watchdog_thread_ = std::thread(&PlatformManager::leaseWatchdogLoop, this);

	std::cout << GREEN << "PlatformManager: Started" << RESET << std::endl;
	return true;
}

void PlatformManager::stop()
{
	if (!running_) return;

	running_ = false;

	// If in a lease, clean up
	if (lease_manager_.hasActiveLease()) {
		lease_manager_.endLease();
		switchToSelfMining();
	}

	// Wait for threads
	if (heartbeat_thread_.joinable()) heartbeat_thread_.join();
	if (watchdog_thread_.joinable()) watchdog_thread_.join();

	// Report offline status and disconnect
	reporter_.sendStatusUpdate("offline");
	mqtt_->disconnect();

	transitionTo(PlatformState::IDLE);
	std::cout << "PlatformManager: Stopped" << std::endl;
}

PlatformState PlatformManager::getState() const
{
	return state_.load();
}

bool PlatformManager::isRunning() const
{
	return running_;
}

void PlatformManager::onBlockFound(const std::string& hash,
								   const std::string& key,
								   const std::string& account,
								   size_t attempts,
								   float hashrate)
{
	if (state_ != PlatformState::MINING) return;

	auto lease = lease_manager_.getLease();
	if (!lease.has_value()) return;

	lease_manager_.recordBlock();
	reporter_.sendBlockFound(lease->lease_id, hash, key, account, attempts, hashrate);
}

void PlatformManager::setStateChangeCallback(StateChangeCallback cb)
{
	std::lock_guard<std::mutex> lock(cb_mutex_);
	state_change_cb_ = std::move(cb);
}

// --- State Transitions ---

void PlatformManager::transitionTo(PlatformState new_state)
{
	PlatformState old_state = state_.exchange(new_state);
	if (old_state == new_state) return;

	std::cout << "PlatformManager: " << platformStateToString(old_state)
			  << " -> " << platformStateToString(new_state) << std::endl;

	// Report state change to platform
	if (mqtt_->isConnected()) {
		auto lease = lease_manager_.getLease();
		std::string lease_id = lease.has_value() ? lease->lease_id : "";
		reporter_.sendStatusUpdate(platformStateToString(new_state), lease_id);
	}

	// Invoke external callback
	{
		std::lock_guard<std::mutex> lock(cb_mutex_);
		if (state_change_cb_) {
			state_change_cb_(old_state, new_state);
		}
	}
}

// --- MQTT Message Dispatch ---

void PlatformManager::onMessage(const std::string& topic, const std::string& payload)
{
	try {
		auto msg = nlohmann::json::parse(payload);
		std::string command = msg.value("command", "");
		std::cerr << "[DEBUG] MQTT message: topic=" << topic << " command=" << command << std::endl;

		if (command == "register_ack") {
			handleRegisterAck(msg);
		} else if (command == "assign_task") {
			handleAssignTask(msg);
		} else if (command == "release") {
			handleRelease(msg);
		} else {
			handleControl(msg);
		}
	} catch (const nlohmann::json::exception& e) {
		std::cerr << YELLOW << "PlatformManager: Invalid JSON message: " << e.what() << RESET << std::endl;
	}
}

void PlatformManager::handleRegisterAck(const nlohmann::json& msg)
{
	bool accepted = msg.value("accepted", false);
	if (accepted) {
		std::cout << GREEN << "PlatformManager: Registration accepted by platform" << RESET << std::endl;
		if (state_ != PlatformState::AVAILABLE) {
			transitionTo(PlatformState::AVAILABLE);
		}
	} else {
		std::string reason = msg.value("reason", "unknown");
		std::cerr << RED << "PlatformManager: Registration rejected: " << reason << RESET << std::endl;
		transitionTo(PlatformState::ERROR);
	}
}

void PlatformManager::handleAssignTask(const nlohmann::json& msg)
{
	std::cerr << "[DEBUG] handleAssignTask called, state=" << platformStateToString(state_) << std::endl;
	if (state_ != PlatformState::AVAILABLE) {
		std::cerr << YELLOW << "PlatformManager: Received assign_task but not AVAILABLE (state="
				  << platformStateToString(state_) << ")" << RESET << std::endl;
		return;
	}

	std::string lease_id = msg.at("lease_id").get<std::string>();
	std::string consumer_id = msg.at("consumer_id").get<std::string>();
	std::string consumer_address = msg.at("consumer_address").get<std::string>();
	std::string prefix = msg.value("prefix", "");
	int duration_sec = msg.value("duration_sec", 3600);

	std::cerr << "[DEBUG] assign_task parsed: lease=" << lease_id << " prefix=" << prefix << " len=" << prefix.length() << std::endl;

	// Validate prefix length
	if (!prefix.empty() && prefix.length() != PLATFORM_PREFIX_LENGTH) {
		std::cerr << RED << "PlatformManager: Invalid prefix length: " << prefix.length() << RESET << std::endl;
		transitionTo(PlatformState::ERROR);
		return;
	}

	transitionTo(PlatformState::LEASED);
	std::cerr << "[DEBUG] transitioned to LEASED, calling startLease" << std::endl;

	// Start the lease
	if (!lease_manager_.startLease(lease_id, consumer_id, consumer_address, prefix, duration_sec)) {
		std::cerr << RED << "PlatformManager: Failed to start lease" << RESET << std::endl;
		transitionTo(PlatformState::ERROR);
		return;
	}

	std::cerr << "[DEBUG] startLease succeeded, switching to platform mining" << std::endl;

	// Switch mining to platform mode
	MiningContext ctx = lease_manager_.toMiningContext();
	switchToPlatformMining(ctx);

	std::cerr << "[DEBUG] switchToPlatformMining done, transitioning to MINING" << std::endl;
	transitionTo(PlatformState::MINING);
	std::cerr << "[DEBUG] handleAssignTask complete, state=MINING" << std::endl;
}

void PlatformManager::handleRelease(const nlohmann::json& msg)
{
	std::string lease_id = msg.value("lease_id", "");

	auto current = lease_manager_.getLease();
	if (!current.has_value()) {
		std::cout << YELLOW << "PlatformManager: Release received but no active lease" << RESET << std::endl;
		return;
	}

	if (!lease_id.empty() && current->lease_id != lease_id) {
		std::cerr << YELLOW << "PlatformManager: Release for wrong lease_id: " << lease_id
				  << " (current: " << current->lease_id << ")" << RESET << std::endl;
		return;
	}

	std::cout << "PlatformManager: Releasing lease " << current->lease_id << std::endl;

	transitionTo(PlatformState::COMPLETED);

	lease_manager_.endLease();
	switchToSelfMining();

	transitionTo(PlatformState::AVAILABLE);
}

void PlatformManager::handleControl(const nlohmann::json& msg)
{
	std::string action = msg.value("action", "");

	if (action == "pause") {
		std::cout << "PlatformManager: Pause requested" << std::endl;
		if (lease_manager_.hasActiveLease()) {
			lease_manager_.endLease();
			switchToSelfMining();
		}
		transitionTo(PlatformState::IDLE);
	} else if (action == "resume") {
		std::cout << "PlatformManager: Resume requested" << std::endl;
		if (state_ == PlatformState::IDLE) {
			reporter_.sendRegistration(eth_address_, gpus_);
			transitionTo(PlatformState::AVAILABLE);
		}
	} else if (action == "shutdown") {
		std::cout << "PlatformManager: Shutdown requested" << std::endl;
		stop();
	}
}

// --- Heartbeat ---

void PlatformManager::heartbeatLoop()
{
	while (running_) {
		// Sleep in small increments so we can exit promptly
		for (int i = 0; i < HEARTBEAT_INTERVAL_SEC && running_; ++i) {
			std::this_thread::sleep_for(std::chrono::seconds(1));
		}
		if (!running_) break;

		// Gather stats from global state
		float total_hashrate = 0;
		int active_gpus = 0;
		{
			std::lock_guard<std::mutex> lock(globalGpuInfosMutex);
			for (const auto& [idx, pair] : globalGpuInfos) {
				total_hashrate += pair.first.hashrate;
				active_gpus++;
			}
		}
		int accepted_blocks = globalNormalBlockCount + globalSuperBlockCount;

		reporter_.sendHeartbeat(total_hashrate, active_gpus, accepted_blocks);
	}
}

// --- Lease Watchdog ---

void PlatformManager::leaseWatchdogLoop()
{
	while (running_) {
		for (int i = 0; i < WATCHDOG_INTERVAL_SEC && running_; ++i) {
			std::this_thread::sleep_for(std::chrono::seconds(1));
		}
		if (!running_) break;

		// Check if the current lease has expired
		if (state_ == PlatformState::MINING && lease_manager_.isExpired()) {
			std::cout << YELLOW << "PlatformManager: Lease expired" << RESET << std::endl;

			transitionTo(PlatformState::COMPLETED);

			lease_manager_.endLease();
			switchToSelfMining();

			transitionTo(PlatformState::AVAILABLE);
		}

		// Attempt recovery from error state
		if (state_ == PlatformState::ERROR) {
			std::cout << "PlatformManager: Attempting recovery..." << std::endl;
			transitionTo(PlatformState::IDLE);
			if (mqtt_->isConnected()) {
				reporter_.sendRegistration(eth_address_, gpus_);
				transitionTo(PlatformState::AVAILABLE);
			} else {
				mqtt_->connect();
			}
		}
	}
}

// --- Mining Mode Switching ---

void PlatformManager::switchToSelfMining()
{
	MiningContext ctx;
	ctx.mode = MiningMode::SELF_MINING;
	ctx.address = globalUserAddress;
	MiningCoordinator::getInstance().updateContext(ctx);
	std::cout << "PlatformManager: Switched to self-mining" << std::endl;
}

void PlatformManager::switchToPlatformMining(const MiningContext& ctx)
{
	MiningCoordinator::getInstance().updateContext(ctx);
	std::cout << GREEN << "PlatformManager: Switched to platform mining for "
			  << ctx.address << RESET << std::endl;
}
