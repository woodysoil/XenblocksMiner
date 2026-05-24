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

} // namespace

HashApiResult CpuHashBackend::runBatch(const HashApiRequest& request)
{
    HashApiResult result;
    result.request_id = request.request_id;
    result.algorithm = request.algorithm;
    result.backend = request.backend == "reference" ? "reference" : "cpu";
    result.device_id = request.device_id;
    result.batch_size = request.batch_size;

    const auto errors = validateRequest(request);
    if (!errors.empty()) {
        result.error = joinErrors(errors);
        return result;
    }
    if (request.backend == "cuda") {
        result.error = "cuda backend is not available in CpuHashBackend";
        return result;
    }
    if (request.difficulty < kMinArgon2CpuDifficulty) {
        result.error = "cpu/reference difficulty must be at least 8";
        return result;
    }

    const auto start = std::chrono::steady_clock::now();
    const std::string salt = normalizeHex(request.salt_hex);
    const std::string prefix = normalizeHex(request.key_prefix);
    const std::string fixed_key = normalizeHex(request.key);
    const bool single_key = !fixed_key.empty();
    const std::size_t attempts = single_key ? 1 : request.batch_size;

    try {
        Argon2idHasher hasher(1, request.difficulty, 1, salt, kDefaultHashLength);
        RandomHexKeyGenerator key_generator(prefix, kHashApiKeyLength);

        for (std::size_t i = 0; i < attempts; ++i) {
            const std::string key = single_key ? fixed_key : key_generator.nextRandomKey();
            const std::string hash = hasher.generateHash(key);
            if (single_key) {
                result.hash = hash;
            }
            appendMatches(request, result, key, hash, i);
        }

        result.ok = true;
        result.attempts = attempts;
    } catch (const std::exception& ex) {
        result.error = ex.what();
    }

    const auto end = std::chrono::steady_clock::now();
    result.elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    if (result.elapsed_ms > 0.0 && result.attempts > 0) {
        result.hashrate = static_cast<double>(result.attempts) / (result.elapsed_ms / 1000.0);
    }

    return result;
}

} // namespace hashapi
