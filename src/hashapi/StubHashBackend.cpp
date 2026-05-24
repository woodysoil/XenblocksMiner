#include "CpuHashBackend.h"

#include "HashApiValidation.h"

#include <chrono>
#include <functional>
#include <iomanip>
#include <sstream>

namespace hashapi {
namespace {

std::string pseudoHash(const std::string& salt, const std::string& key, std::uint32_t difficulty)
{
    const std::string input = salt + ":" + key + ":" + std::to_string(difficulty);
    std::hash<std::string> hasher;
    std::ostringstream out;
    out << "stub$argon2id-xen$";
    for (int i = 0; i < 8; ++i) {
        out << std::hex << std::setw(16) << std::setfill('0')
            << hasher(input + ":" + std::to_string(i));
    }
    return out.str();
}

std::string makeKey(const std::string& prefix, std::size_t index)
{
    std::ostringstream suffix;
    const std::size_t suffix_length = kHashApiKeyLength - prefix.size();
    suffix << std::hex << std::setw(static_cast<int>(suffix_length)) << std::setfill('0') << index;
    std::string suffix_text = suffix.str();
    if (suffix_text.size() > suffix_length) {
        suffix_text = suffix_text.substr(suffix_text.size() - suffix_length);
    }
    return prefix + suffix_text;
}

void appendMatchIfNeeded(const HashApiRequest& request,
                         HashApiResult& result,
                         const std::string& key,
                         const std::string& hash,
                         std::size_t attempt_index)
{
    if (hash.find(request.target_pattern) != std::string::npos) {
        result.matches.push_back({
            key,
            hash,
            request.target_pattern,
            attempt_index,
            false,
        });
    }
}

} // namespace

HashApiResult CpuHashBackend::runBatch(const HashApiRequest& request)
{
    HashApiResult result;
    result.request_id = request.request_id;
    result.algorithm = request.algorithm;
    result.backend = request.backend == "reference" ? "reference-stub" : "cpu-stub";
    result.device_id = request.device_id;
    result.batch_size = request.batch_size;

    const auto errors = validateRequest(request);
    if (!errors.empty()) {
        result.error = joinErrors(errors);
        return result;
    }
    if (request.backend == "cuda") {
        result.error = "cuda backend is not available in the stub backend";
        return result;
    }

    const auto start = std::chrono::steady_clock::now();
    const std::string salt = normalizeHex(request.salt_hex);
    const std::string prefix = normalizeHex(request.key_prefix);
    const std::string fixed_key = normalizeHex(request.key);
    const bool single_key = !fixed_key.empty();
    const std::size_t attempts = single_key ? 1 : request.batch_size;

    for (std::size_t i = 0; i < attempts; ++i) {
        const std::string key = single_key ? fixed_key : makeKey(prefix, i);
        const std::string hash = pseudoHash(salt, key, request.difficulty);
        if (single_key) {
            result.hash = hash;
        }
        appendMatchIfNeeded(request, result, key, hash, i);
    }

    result.ok = true;
    result.attempts = attempts;

    const auto end = std::chrono::steady_clock::now();
    result.elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    if (result.elapsed_ms > 0.0 && result.attempts > 0) {
        result.hashrate = static_cast<double>(result.attempts) / (result.elapsed_ms / 1000.0);
    }

    return result;
}

} // namespace hashapi
