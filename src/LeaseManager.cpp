#include "LeaseManager.h"

#include <iostream>

bool LeaseManager::startLease(const std::string& lease_id,
							  const std::string& consumer_id,
							  const std::string& consumer_address,
							  const std::string& prefix,
							  int duration_sec)
{
	std::lock_guard<std::mutex> lock(mutex_);

	if (current_lease_.has_value()) {
		std::cerr << RED << "LeaseManager: Cannot start lease " << lease_id
				  << " - already in lease " << current_lease_->lease_id << RESET << std::endl;
		return false;
	}

	LeaseInfo lease;
	lease.lease_id = lease_id;
	lease.consumer_id = consumer_id;
	lease.consumer_address = consumer_address;
	lease.prefix = prefix;
	lease.duration_sec = duration_sec;
	lease.start_time = std::chrono::steady_clock::now();
	lease.blocks_found = 0;

	current_lease_ = lease;

	std::cout << GREEN << "LeaseManager: Lease started - " << lease_id
			  << " for " << consumer_address
			  << " (" << duration_sec << "s)" << RESET << std::endl;
	return true;
}

bool LeaseManager::endLease()
{
	std::lock_guard<std::mutex> lock(mutex_);

	if (!current_lease_.has_value()) {
		return false;
	}

	std::cout << "LeaseManager: Lease ended - " << current_lease_->lease_id
			  << " (blocks found: " << current_lease_->blocks_found << ")" << std::endl;

	current_lease_.reset();
	return true;
}

bool LeaseManager::hasActiveLease() const
{
	std::lock_guard<std::mutex> lock(mutex_);
	return current_lease_.has_value() && !isExpiredInternal();
}

bool LeaseManager::isExpired() const
{
	std::lock_guard<std::mutex> lock(mutex_);
	return isExpiredInternal();
}

void LeaseManager::recordBlock()
{
	std::lock_guard<std::mutex> lock(mutex_);
	if (current_lease_.has_value()) {
		current_lease_->blocks_found++;
	}
}

std::optional<LeaseInfo> LeaseManager::getLease() const
{
	std::lock_guard<std::mutex> lock(mutex_);
	return current_lease_;
}

int LeaseManager::remainingSeconds() const
{
	std::lock_guard<std::mutex> lock(mutex_);
	if (!current_lease_.has_value()) return 0;

	auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
		std::chrono::steady_clock::now() - current_lease_->start_time).count();
	int remaining = current_lease_->duration_sec - static_cast<int>(elapsed);
	return remaining > 0 ? remaining : 0;
}

MiningContext LeaseManager::toMiningContext() const
{
	std::lock_guard<std::mutex> lock(mutex_);

	if (!current_lease_.has_value()) {
		// Return self-mining context
		MiningContext ctx;
		ctx.mode = MiningMode::SELF_MINING;
		ctx.address = globalUserAddress;
		return ctx;
	}

	MiningContext ctx;
	ctx.mode = MiningMode::PLATFORM_MINING;
	ctx.address = current_lease_->consumer_address;
	ctx.prefix = current_lease_->prefix;
	ctx.consumer_id = current_lease_->consumer_id;
	ctx.lease_id = current_lease_->lease_id;
	return ctx;
}

bool LeaseManager::isExpiredInternal() const
{
	if (!current_lease_.has_value()) return true;
	if (current_lease_->duration_sec <= 0) return false; // 0 means no expiry

	auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
		std::chrono::steady_clock::now() - current_lease_->start_time).count();
	return elapsed >= current_lease_->duration_sec;
}
