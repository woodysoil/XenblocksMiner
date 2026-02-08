#pragma once

#include <shared_mutex>
#include <string>
#include "MiningCommon.h"

class MiningCoordinator
{
public:
	static MiningCoordinator& getInstance();

	MiningContext getContext() const;
	void updateContext(const MiningContext& ctx);
	MiningMode getMode() const;

private:
	MiningCoordinator() = default;
	MiningCoordinator(const MiningCoordinator&) = delete;
	MiningCoordinator& operator=(const MiningCoordinator&) = delete;

	mutable std::shared_mutex mutable_mutex_;
	MiningContext context_;
};
