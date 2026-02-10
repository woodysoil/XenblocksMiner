#pragma once

#include <string>
#include <memory>
#include <atomic>
#include <thread>
#include <mutex>
#include <functional>
#include <vector>
#include "MqttClient.h"
#include "WorkerReporter.h"
#include "LeaseManager.h"
#include "MiningCoordinator.h"
#include "MiningCommon.h"

// Platform states for the hashpower marketplace
enum class PlatformState {
	IDLE,       // Not connected to platform
	AVAILABLE,  // Registered and waiting for lease assignment
	LEASED,     // Lease assigned, preparing to mine
	MINING,     // Actively mining for a consumer
	COMPLETED,  // Lease completed, transitioning back
	ERROR       // Error state, will attempt recovery
};

const char* platformStateToString(PlatformState state);

class PlatformManager
{
public:
	PlatformManager(const std::string& broker_uri,
					const std::string& eth_address,
					const std::vector<gpuInfo>& gpus);
	~PlatformManager();

	// Lifecycle
	bool start();
	void stop();

	// State queries
	PlatformState getState() const;
	bool isRunning() const;

	// Called by submitCallback when a block is found during platform mining
	void onBlockFound(const std::string& hash,
					  const std::string& key,
					  const std::string& account,
					  size_t attempts,
					  float hashrate);

	// Access to lease manager for external queries
	const LeaseManager& getLeaseManager() const { return lease_manager_; }

	// State change callback (for external monitoring)
	using StateChangeCallback = std::function<void(PlatformState old_state, PlatformState new_state)>;
	void setStateChangeCallback(StateChangeCallback cb);

private:
	// State transitions
	void transitionTo(PlatformState new_state);

	// MQTT message handler (dispatches to state-specific handlers)
	void onMessage(const std::string& topic, const std::string& payload);

	// Command handlers (from platform via MQTT)
	void handleRegisterAck(const nlohmann::json& msg);
	void handleAssignTask(const nlohmann::json& msg);
	void handleRelease(const nlohmann::json& msg);
	void handleControl(const nlohmann::json& msg);
	void handleSetConfig(const nlohmann::json& msg);

	// Heartbeat thread
	void heartbeatLoop();

	// Lease expiry checker thread
	void leaseWatchdogLoop();

	// Switch mining mode via MiningCoordinator
	void switchToSelfMining();
	void switchToPlatformMining(const MiningContext& ctx);

	// State
	std::atomic<PlatformState> state_{PlatformState::IDLE};
	std::atomic<bool> running_{false};

	// Components
	std::shared_ptr<MqttClient> mqtt_;
	WorkerReporter reporter_;
	LeaseManager lease_manager_;

	// Config
	std::string eth_address_;
	std::vector<gpuInfo> gpus_;

	// Threads
	std::thread heartbeat_thread_;
	std::thread watchdog_thread_;

	// Callback
	StateChangeCallback state_change_cb_;
	std::mutex cb_mutex_;

	static constexpr int HEARTBEAT_INTERVAL_SEC = 30;
	static constexpr int WATCHDOG_INTERVAL_SEC = 5;
	static constexpr int ERROR_RECOVERY_DELAY_SEC = 10;
};
