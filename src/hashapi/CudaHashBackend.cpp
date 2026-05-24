#include "CudaHashBackend.h"

#include "HashApiEncoding.h"
#include "HashApiMatching.h"
#include "HashApiValidation.h"
#include "../ComputeBackend.h"
#include "../RandomHexKeyGenerator.h"
#include "../argon2-common.h"
#include "../argon2params.h"

#include <chrono>
#include <exception>
#include <stdexcept>
#include <utility>

namespace hashapi {
namespace {

void fillPasswordBlock(ComputeBackend& backend,
                       const Argon2Params& params,
                       std::size_t index,
                       const std::string& password)
{
    params.fillFirstBlocks(backend.getInputMemory(index), password.c_str(), password.size());
}

std::string finalizeHash(ComputeBackend& backend,
                         const Argon2Params& params,
                         std::size_t index)
{
    std::uint8_t buffer[kDefaultHashLength];
    params.finalize(buffer, backend.getOutputMemory(index));
    return base64Encode(buffer, kDefaultHashLength);
}

} // namespace

CudaHashBackend::CudaHashBackend(ComputeBackend& backend)
    : backend_(&backend)
{
}

CudaHashBackend::CudaHashBackend(std::unique_ptr<ComputeBackend> backend)
    : backend_(backend.get()), owned_backend_(std::move(backend))
{
    if (backend_ == nullptr) {
        throw std::invalid_argument("cuda backend cannot be null");
    }
}

CudaHashBackend::~CudaHashBackend() = default;

ComputeBackend& CudaHashBackend::backend()
{
    if (backend_ == nullptr) {
        throw std::runtime_error("cuda backend is not initialized");
    }
    return *backend_;
}

const ComputeBackend& CudaHashBackend::backend() const
{
    if (backend_ == nullptr) {
        throw std::runtime_error("cuda backend is not initialized");
    }
    return *backend_;
}

void CudaHashBackend::ensureInitialized(ComputeBackend& backend,
                                        const Argon2Params& params,
                                        std::size_t batch_size,
                                        std::uint32_t difficulty)
{
    const auto segment_blocks = params.getSegmentBlocks();
    if (initialized_ &&
        initialized_batch_size_ == batch_size &&
        initialized_difficulty_ == difficulty &&
        initialized_segment_blocks_ == segment_blocks) {
        return;
    }

    backend.init(batch_size, argon2::ARGON2_ID, argon2::ARGON2_VERSION_13,
                 1, 1, segment_blocks);
    initialized_ = true;
    initialized_batch_size_ = batch_size;
    initialized_difficulty_ = difficulty;
    initialized_segment_blocks_ = segment_blocks;
}

HashApiResult CudaHashBackend::runBatch(const HashApiRequest& request)
{
    HashApiResult result;
    result.request_id = request.request_id;
    result.algorithm = request.algorithm;
    result.backend = "cuda";
    result.device_id = request.device_id;
    result.batch_size = request.batch_size;

    const auto errors = validateRequest(request);
    if (!errors.empty()) {
        result.error = joinErrors(errors);
        return result;
    }
    if (request.backend != "cuda") {
        result.error = "CudaHashBackend requires backend=cuda";
        return result;
    }

    const auto start = std::chrono::steady_clock::now();
    const std::string salt = normalizeHex(request.salt_hex);
    const std::string prefix = normalizeHex(request.key_prefix);
    const std::string fixed_key = normalizeHex(request.key);
    const bool single_key = !fixed_key.empty();
    const std::size_t attempts = single_key ? 1 : request.batch_size;

    try {
        auto& compute_backend = backend();
        compute_backend.activate();
        const auto device_info = compute_backend.getDeviceInfo();
        result.device_id = device_info.index;

        Argon2Params params(argon2::ARGON2_ID, argon2::ARGON2_VERSION_13,
                            kDefaultHashLength, salt, nullptr, 0, nullptr, 0,
                            1, request.difficulty, 1);
        ensureInitialized(compute_backend, params, attempts, request.difficulty);

        password_storage_.clear();
        password_storage_.reserve(attempts);
        RandomHexKeyGenerator key_generator(prefix, kHashApiKeyLength);

        for (std::size_t i = 0; i < attempts; ++i) {
            const std::string key = single_key ? fixed_key : key_generator.nextRandomKey();
            fillPasswordBlock(compute_backend, params, i, key);
            password_storage_.push_back(key);
        }

        compute_backend.run();
        compute_backend.finish();

        for (std::size_t i = 0; i < attempts; ++i) {
            const std::string hash = finalizeHash(compute_backend, params, i);
            const std::string& key = password_storage_[i];
            if (single_key) {
                result.hash = hash;
            }
            appendMatches(request, result, key, hash, i);
        }

        result.ok = true;
        result.attempts = attempts;
        result.batch_size = attempts;
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
