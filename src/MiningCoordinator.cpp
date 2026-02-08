#include "MiningCoordinator.h"

MiningCoordinator& MiningCoordinator::getInstance()
{
	static MiningCoordinator instance;
	return instance;
}

MiningContext MiningCoordinator::getContext() const
{
	std::shared_lock<std::shared_mutex> lock(mutex_);
	return context_;
}

MiningMode MiningCoordinator::getMode() const
{
	std::shared_lock<std::shared_mutex> lock(mutex_);
	return context_.mode;
}

bool MiningCoordinator::isSelfMining() const
{
	return getMode() == MiningMode::SELF_MINING;
}

bool MiningCoordinator::isPlatformMining() const
{
	return getMode() == MiningMode::PLATFORM_MINING;
}

void MiningCoordinator::switchToSelfMining()
{
	std::unique_lock<std::shared_mutex> lock(mutex_);
	context_.mode = MiningMode::SELF_MINING;
	context_.address.clear();
	context_.prefix.clear();
	context_.consumer_id.clear();
	context_.lease_id.clear();
}

void MiningCoordinator::switchToPlatformMining(const std::string& address,
	const std::string& prefix,
	const std::string& consumer_id,
	const std::string& lease_id)
{
	std::unique_lock<std::shared_mutex> lock(mutex_);
	context_.mode = MiningMode::PLATFORM_MINING;
	context_.address = address;
	context_.prefix = prefix;
	context_.consumer_id = consumer_id;
	context_.lease_id = lease_id;
}
