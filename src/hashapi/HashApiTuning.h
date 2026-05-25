#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace hashapi {

constexpr std::size_t kCudaBatchMemoryReserveBytes = 100ULL * 1024ULL * 1024ULL;

struct CudaBatchSizeDecision {
    std::size_t memory_limited_batch_size = 0;
    std::size_t tuned_batch_size = 0;
    std::size_t selected_batch_size = 0;
    bool explicit_limit_applied = false;
    bool tuned_default_applied = false;
};

std::size_t estimateCudaMemoryBatchLimit(std::size_t free_memory_bytes,
                                         std::uint32_t difficulty,
                                         std::size_t reserve_bytes = kCudaBatchMemoryReserveBytes);

std::size_t recommendedCudaBatchSize(std::uint32_t difficulty);

std::size_t recommendedCudaBatchSizeForDifficultySequence(
    const std::vector<std::uint32_t>& difficulties);

CudaBatchSizeDecision selectCudaBatchSize(std::size_t free_memory_bytes,
                                          std::uint32_t difficulty,
                                          std::size_t explicit_max_batch_size);

CudaBatchSizeDecision selectCudaBatchSizeForDifficultySequence(
    std::size_t free_memory_bytes,
    const std::vector<std::uint32_t>& difficulties,
    std::size_t explicit_max_batch_size);

} // namespace hashapi
