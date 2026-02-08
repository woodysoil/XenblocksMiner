#pragma once

#include <shared_mutex>
#include <string>
#include "MiningCommon.h"

class MiningCoordinator
{
public:
	static MiningCoordinator& getInstance();

	// Read current context snapshot (shared/read lock)
	MiningContext getContext() const;

	// Query current mode (shared/read lock)
	MiningMode getMode() const;
	bool isSelfMining() const;
	bool isPlatformMining() const;

	// Mode switching (exclusive/write lock)
	void switchToSelfMining();
	void switchToPlatformMining(const std::string& address,
		const std::string& prefix,
		const std::string& consumer_id,
		const std::string& lease_id);

private:
	MiningCoordinator() = default;
	MiningCoordinator(const MiningCoordinator&) = delete;
	MiningCoordinator& operator=(const MiningCoordinator&) = delete;

	mutable std::shared_mutex mutex_;
	MiningContext context_;
};
