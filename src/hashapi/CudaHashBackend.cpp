#include "CudaHashBackend.h"

#include "HashApiEncoding.h"
#include "HashApiMatching.h"
#include "HashApiValidation.h"
#include "../ComputeBackend.h"
#include "../RandomHexKeyGenerator.h"
#include "../argon2-common.h"
#include "../argon2params.h"

#include <algorithm>
#include <chrono>
#include <exception>
#include <stdexcept>
#include <thread>
#include <utility>

namespace hashapi {
namespace {

double elapsedMillis(std::chrono::steady_clock::time_point start,
                     std::chrono::steady_clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

constexpr std::size_t kMinParallelFirstBlockAttempts = 8;

void fillPasswordBlock(ComputeBackend& backend,
                       const Argon2Params& params,
                       std::size_t index,
                       const std::string& password)
{
    params.fillFirstBlocks(backend.getInputMemory(index), password.c_str(), password.size());
}

std::size_t firstBlockWorkerCount(std::size_t attempts)
{
    if (attempts < kMinParallelFirstBlockAttempts) {
        return 1;
    }

    const unsigned int hardware_threads = std::thread::hardware_concurrency();
    if (hardware_threads < 2) {
        return 1;
    }

    return std::min<std::size_t>(attempts, hardware_threads);
}

void fillPasswordBlocks(ComputeBackend& backend,
                        const Argon2Params& params,
                        const std::vector<std::string>& passwords)
{
    const std::size_t attempts = passwords.size();
    const std::size_t worker_count = firstBlockWorkerCount(attempts);
    if (worker_count <= 1) {
        for (std::size_t i = 0; i < attempts; ++i) {
            fillPasswordBlock(backend, params, i, passwords[i]);
        }
        return;
    }

    const std::size_t chunk_size = (attempts + worker_count - 1) / worker_count;
    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    for (std::size_t worker = 0; worker < worker_count; ++worker) {
        const std::size_t begin = worker * chunk_size;
        const std::size_t end = std::min(attempts, begin + chunk_size);
        if (begin >= end) {
            break;
        }
        workers.emplace_back([&backend, &params, &passwords, begin, end]() {
            for (std::size_t i = begin; i < end; ++i) {
                fillPasswordBlock(backend, params, i, passwords[i]);
            }
        });
    }

    for (std::thread& worker : workers) {
        worker.join();
    }
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
                                        std::size_t batch_size)
{
    const auto segment_blocks = params.getSegmentBlocks();
    if (initialized_ &&
        initialized_batch_size_ == batch_size &&
        initialized_segment_blocks_ == segment_blocks) {
        return;
    }

    backend.init(batch_size, argon2::ARGON2_ID, argon2::ARGON2_VERSION_13,
                 1, 1, segment_blocks);
    initialized_ = true;
    initialized_batch_size_ = batch_size;
    initialized_segment_blocks_ = segment_blocks;
}

HashApiResult CudaHashBackend::runBatch(const HashApiRequest& request)
{
    const auto total_start = std::chrono::steady_clock::now();
    HashApiResult result;
    result.request_id = request.request_id;
    result.algorithm = request.algorithm;
    result.backend = "cuda";
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
    if (request.backend != "cuda") {
        result.error = "CudaHashBackend requires backend=cuda";
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

        auto& compute_backend = backend();
        compute_backend.activate();
        const auto device_info = compute_backend.getDeviceInfo();
        result.device_id = device_info.index;

        Argon2Params params(argon2::ARGON2_ID, argon2::ARGON2_VERSION_13,
                            kDefaultHashLength, salt, nullptr, 0, nullptr, 0,
                            1, request.difficulty, 1);
        ensureInitialized(compute_backend, params, attempts);
        result.timings.setup_ms = elapsedMillis(setup_start, std::chrono::steady_clock::now());

        const auto input_start = std::chrono::steady_clock::now();
        password_storage_.clear();
        password_storage_.reserve(attempts);
        RandomHexKeyGenerator key_generator(prefix, kHashApiKeyLength);

        if (firstBlockWorkerCount(attempts) <= 1) {
            for (std::size_t i = 0; i < attempts; ++i) {
                const auto keygen_start = std::chrono::steady_clock::now();
                const std::string key = single_key ? fixed_key : key_generator.nextRandomKey();
                const auto keygen_end = std::chrono::steady_clock::now();
                result.timings.keygen_ms += elapsedMillis(keygen_start, keygen_end);

                const auto first_block_start = std::chrono::steady_clock::now();
                fillPasswordBlock(compute_backend, params, i, key);
                const auto first_block_end = std::chrono::steady_clock::now();
                result.timings.first_block_ms += elapsedMillis(first_block_start, first_block_end);
                password_storage_.push_back(key);
            }
        } else {
            const auto keygen_start = std::chrono::steady_clock::now();
            for (std::size_t i = 0; i < attempts; ++i) {
                password_storage_.push_back(single_key ? fixed_key : key_generator.nextRandomKey());
            }
            result.timings.keygen_ms = elapsedMillis(keygen_start, std::chrono::steady_clock::now());

            const auto first_block_start = std::chrono::steady_clock::now();
            fillPasswordBlocks(compute_backend, params, password_storage_);
            result.timings.first_block_ms = elapsedMillis(first_block_start, std::chrono::steady_clock::now());
        }
        result.timings.input_ms = elapsedMillis(input_start, std::chrono::steady_clock::now());

        const auto compute_start = std::chrono::steady_clock::now();
        compute_backend.run();
        compute_backend.finish();
        result.timings.compute_ms = elapsedMillis(compute_start, std::chrono::steady_clock::now());

        const auto finalize_start = std::chrono::steady_clock::now();
        for (std::size_t i = 0; i < attempts; ++i) {
            const std::string hash = finalizeHash(compute_backend, params, i);
            const std::string& key = password_storage_[i];
            if (single_key) {
                result.hash = hash;
            }
            appendMatches(request, result, key, hash, i);
        }
        result.timings.finalize_ms = elapsedMillis(finalize_start, std::chrono::steady_clock::now());

        result.ok = true;
        result.attempts = attempts;
        result.batch_size = attempts;
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
