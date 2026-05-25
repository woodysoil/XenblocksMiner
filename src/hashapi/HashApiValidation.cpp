#include "HashApiValidation.h"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <unordered_set>

namespace hashapi {
namespace {

bool isSupportedAlgorithm(const std::string& algorithm)
{
    return algorithm == "argon2id-xen";
}

bool isSupportedBackend(const std::string& backend)
{
    static const std::unordered_set<std::string> backends = {
        "cpu", "reference", "cuda"
    };
    return backends.find(backend) != backends.end();
}

} // namespace

bool isHexString(const std::string& value)
{
    return std::all_of(value.begin(), value.end(), [](unsigned char ch) {
        return std::isxdigit(ch) != 0;
    });
}

std::string normalizeHex(const std::string& value)
{
    std::string normalized = value;
    if (normalized.size() >= 2 && normalized[0] == '0' &&
        (normalized[1] == 'x' || normalized[1] == 'X')) {
        normalized = normalized.substr(2);
    }
    std::transform(normalized.begin(), normalized.end(), normalized.begin(),
        [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return normalized;
}

std::vector<std::string> validateRequest(const HashApiRequest& request)
{
    std::vector<std::string> errors;

    if (!isSupportedAlgorithm(request.algorithm)) {
        errors.push_back("unsupported algorithm: " + request.algorithm);
    }

    if (!isSupportedBackend(request.backend)) {
        errors.push_back("unsupported backend: " + request.backend);
    }

    const std::string salt = normalizeHex(request.salt_hex);
    if (salt.empty()) {
        errors.push_back("salt_hex is required");
    } else {
        if (salt.size() % 2 != 0) {
            errors.push_back("salt_hex must contain an even number of hex characters");
        }
        if (salt.size() < 16) {
            errors.push_back("salt_hex must be at least 16 hex characters");
        }
        if (!isHexString(salt)) {
            errors.push_back("salt_hex must contain only hex characters");
        }
    }

    const std::string prefix = normalizeHex(request.key_prefix);
    if (!prefix.empty()) {
        if (prefix.size() > kHashApiKeyLength) {
            errors.push_back("key_prefix cannot exceed 64 hex characters");
        }
        if (!isHexString(prefix)) {
            errors.push_back("key_prefix must contain only hex characters");
        }
    }

    const std::string key = normalizeHex(request.key);
    if (!key.empty()) {
        if (key.size() != kHashApiKeyLength) {
            errors.push_back("key must contain exactly 64 hex characters");
        }
        if (!isHexString(key)) {
            errors.push_back("key must contain only hex characters");
        }
        if (!prefix.empty() && key.rfind(prefix, 0) != 0) {
            errors.push_back("key must start with key_prefix when both are provided");
        }
    }

    if (request.target_pattern.empty()) {
        errors.push_back("target_pattern is required");
    }
    if (request.target_pattern.size() > kMaxTargetPatternLength) {
        errors.push_back("target_pattern is too long");
    }

    if (request.difficulty == 0) {
        errors.push_back("difficulty must be greater than zero");
    }

    if (request.batch_size == 0) {
        errors.push_back("batch_size must be greater than zero");
    }
    if (request.backend == "cpu" || request.backend == "reference") {
        if (request.batch_size > kMaxCpuBatchSize) {
            errors.push_back("cpu batch_size exceeds safe limit");
        }
    }

    if (request.device_id < 0) {
        errors.push_back("device_id must be non-negative");
    }

    if (request.gpu_first_blocks && request.backend != "cuda") {
        errors.push_back("gpu_first_blocks requires backend=cuda");
    }

    return errors;
}

bool isValidRequest(const HashApiRequest& request)
{
    return validateRequest(request).empty();
}

std::string joinErrors(const std::vector<std::string>& errors)
{
    std::ostringstream out;
    for (std::size_t i = 0; i < errors.size(); ++i) {
        if (i > 0) {
            out << "; ";
        }
        out << errors[i];
    }
    return out.str();
}

} // namespace hashapi
