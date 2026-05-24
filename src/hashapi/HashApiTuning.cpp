#include "HashApiTuning.h"

#include <algorithm>

namespace hashapi {

std::size_t estimateCudaMemoryBatchLimit(std::size_t free_memory_bytes,
                                         std::uint32_t difficulty,
                                         std::size_t reserve_bytes)
{
    if (difficulty == 0 || free_memory_bytes <= reserve_bytes) {
        return 0;
    }

    const double available_bytes = static_cast<double>(free_memory_bytes - reserve_bytes);
    const double bytes_per_attempt = static_cast<double>(difficulty) * 1024.0 * 1.001;
    if (bytes_per_attempt <= 0.0) {
        return 0;
    }

    return static_cast<std::size_t>(available_bytes / bytes_per_attempt);
}

std::size_t recommendedCudaBatchSize(std::uint32_t difficulty)
{
    if (difficulty <= 1) {
        return 2048;
    }
    if (difficulty <= 8) {
        return 3072;
    }
    if (difficulty <= 64) {
        return 3072;
    }
    return 0;
}

CudaBatchSizeDecision selectCudaBatchSize(std::size_t free_memory_bytes,
                                          std::uint32_t difficulty,
                                          std::size_t explicit_max_batch_size)
{
    CudaBatchSizeDecision decision;
    decision.memory_limited_batch_size = estimateCudaMemoryBatchLimit(free_memory_bytes, difficulty);
    if (decision.memory_limited_batch_size == 0) {
        return decision;
    }

    if (explicit_max_batch_size > 0) {
        decision.selected_batch_size = std::min(decision.memory_limited_batch_size, explicit_max_batch_size);
        decision.explicit_limit_applied = true;
        return decision;
    }

    decision.tuned_batch_size = recommendedCudaBatchSize(difficulty);
    if (decision.tuned_batch_size > 0) {
        decision.selected_batch_size = std::min(decision.memory_limited_batch_size, decision.tuned_batch_size);
        decision.tuned_default_applied = true;
        return decision;
    }

    decision.selected_batch_size = decision.memory_limited_batch_size;
    return decision;
}

} // namespace hashapi
