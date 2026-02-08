#pragma once

#include <string>
#include <mutex>
#include <chrono>
#include <optional>
#include "MiningCommon.h"

struct LeaseInfo {
	std::string lease_id;
	std::string consumer_id;
	std::string consumer_address;  // Ethereum address to mine for
	std::string prefix;            // 16-char hex prefix for key generation
	int duration_sec = 0;          // Lease duration in seconds
	std::chrono::steady_clock::time_point start_time;
	int blocks_found = 0;
};

class LeaseManager
{
public:
	// Start a new lease. Returns false if already in an active lease.
	bool startLease(const std::string& lease_id,
					const std::string& consumer_id,
					const std::string& consumer_address,
					const std::string& prefix,
					int duration_sec);

	// End the current lease. Returns false if no active lease.
	bool endLease();

	// Check if there is an active (non-expired) lease
	bool hasActiveLease() const;

	// Check if the active lease has expired
	bool isExpired() const;

	// Record a block found during the current lease
	void recordBlock();

	// Get current lease info (nullopt if no active lease)
	std::optional<LeaseInfo> getLease() const;

	// Get remaining time in seconds (0 if no lease or expired)
	int remainingSeconds() const;

	// Build a MiningContext from the current lease for MiningCoordinator
	MiningContext toMiningContext() const;

private:
	// Must be called with mutex_ already held
	bool isExpiredInternal() const;

	mutable std::mutex mutex_;
	std::optional<LeaseInfo> current_lease_;
};
