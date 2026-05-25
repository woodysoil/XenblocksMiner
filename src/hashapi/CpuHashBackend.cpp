#include "CpuHashBackend.h"

#include "HashApiMatching.h"
#include "HashApiValidation.h"
#include "../Argon2idHasher.h"
#include "../RandomHexKeyGenerator.h"

#include <chrono>
#include <cstdint>
#include <exception>

namespace hashapi {
namespace {

constexpr std::uint32_t kMinArgon2CpuDifficulty = 8;

double elapsedMillis(std::chrono::steady_clock::time_point start,
                     std::chrono::steady_clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

} // namespace

HashApiResult CpuHashBackend::runBatch(const HashApiRequest& request)
{
    const auto total_start = std::chrono::steady_clock::now();
    HashApiResult result;
    result.request_id = request.request_id;
    result.algorithm = request.algorithm;
    result.backend = request.backend == "reference" ? "reference" : "cpu";
    result.device_id = request.device_id;
    result.batch_size = request.batch_size;

    const auto validation_start = std::chrono::steady_clock::now();
    const auto errors = validateRequest(request);
    result.timings.validation_ms = elapsedMillis(validation_start, std::chrono::steady_clock::now());
    if (!errors.empty()) {
        result.error = joinErrors(errors);
        result.timings.total_ms = elapsedMillis(total_start, std::chrono::steady_clock::now());
        return result;
    }
    if (request.backend == "cuda") {
        result.error = "cuda backend is not available in CpuHashBackend";
        result.timings.total_ms = elapsedMillis(total_start, std::chrono::steady_clock::now());
        return result;
    }
    if (request.difficulty < kMinArgon2CpuDifficulty) {
        result.error = "cpu/reference difficulty must be at least 8";
        result.timings.total_ms = elapsedMillis(total_start, std::chrono::steady_clock::now());
        return result;
    }

    const auto start = std::chrono::steady_clock::now();

    try {
        const auto setup_start = std::chrono::steady_clock::now();
        const std::string salt = normalizeHex(request.salt_hex);
        const std::string prefix = normalizeHex(request.key_prefix);
        const std::string fixed_key = normalizeHex(request.key);
        const bool single_key = !fixed_key.empty();
        const std::size_t attempts = single_key ? 1 : request.batch_size;
        Argon2idHasher hasher(1, request.difficulty, 1, salt, kDefaultHashLength);
        RandomHexKeyGenerator key_generator(prefix, kHashApiKeyLength);
        result.timings.setup_ms = elapsedMillis(setup_start, std::chrono::steady_clock::now());

        const auto compute_start = std::chrono::steady_clock::now();
        for (std::size_t i = 0; i < attempts; ++i) {
            const auto keygen_start = std::chrono::steady_clock::now();
            const std::string key = single_key ? fixed_key : key_generator.nextRandomKey();
            const auto keygen_end = std::chrono::steady_clock::now();
            result.timings.keygen_ms += elapsedMillis(keygen_start, keygen_end);

            const std::string hash = hasher.generateHash(key);
            if (single_key) {
                result.hash = hash;
            }
            appendMatches(request, result, key, hash, i);
        }
        result.timings.compute_ms = elapsedMillis(compute_start, std::chrono::steady_clock::now());

        result.ok = true;
        result.attempts = attempts;
        result.batch_size = attempts;
        result.batch_size_min = attempts;
        result.batch_size_max = attempts;
    } catch (const std::exception& ex) {
        result.error = ex.what();
    }

    const auto end = std::chrono::steady_clock::now();
    result.elapsed_ms = elapsedMillis(start, end);
    result.timings.total_ms = elapsedMillis(total_start, end);
    if (result.elapsed_ms > 0.0 && result.attempts > 0) {
        result.hashrate = static_cast<double>(result.attempts) / (result.elapsed_ms / 1000.0);
    }

    return result;
}

} // namespace hashapi
