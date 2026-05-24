#include "CudaHashBackend.h"

#include "HashApiEncoding.h"
#include "HashApiMatching.h"
#include "HashApiValidation.h"
#include "../ComputeBackend.h"
#include "../RandomHexKeyGenerator.h"
#include "../argon2-common.h"
#include "../argon2params.h"

#include <algorithm>
#include <array>
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
constexpr std::size_t kFinalizeTimingChunkSize = 64;

void fillPasswordBlock(ComputeBackend& backend,
                       const Argon2Params& params,
                       std::size_t index,
                       const std::string& password,
                       Argon2FirstBlockTimings* timings)
{
    if (timings != nullptr) {
        params.fillFirstBlocks(backend.getInputMemory(index), password.c_str(), password.size(), timings);
    } else {
        params.fillFirstBlocks(backend.getInputMemory(index), password.c_str(), password.size());
    }
}

std::size_t firstBlockWorkerCount(std::size_t attempts, std::size_t worker_cap)
{
    if (attempts < kMinParallelFirstBlockAttempts) {
        return 1;
    }

    const unsigned int hardware_threads = std::thread::hardware_concurrency();
    if (hardware_threads < 2) {
        return 1;
    }

    std::size_t worker_count = std::min<std::size_t>(attempts, hardware_threads);
    if (worker_cap > 0) {
        worker_count = std::min(worker_count, worker_cap);
    }
    return std::max<std::size_t>(1, worker_count);
}

std::size_t firstBlockChunkSize(std::size_t attempts, std::size_t worker_count)
{
    if (attempts == 0 || worker_count == 0) {
        return 0;
    }
    return (attempts + worker_count - 1) / worker_count;
}

void fillPasswordBlocks(ComputeBackend& backend,
                        const Argon2Params& params,
                        const std::vector<std::string>& passwords,
                        std::size_t worker_cap,
                        Argon2FirstBlockTimings* timings)
{
    const std::size_t attempts = passwords.size();
    const std::size_t worker_count = firstBlockWorkerCount(attempts, worker_cap);
    if (worker_count <= 1) {
        for (std::size_t i = 0; i < attempts; ++i) {
            fillPasswordBlock(backend, params, i, passwords[i], timings);
        }
        return;
    }

    const std::size_t chunk_size = firstBlockChunkSize(attempts, worker_count);
    std::vector<Argon2FirstBlockTimings> worker_timings(timings == nullptr ? 0 : worker_count);
    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    for (std::size_t worker = 0; worker < worker_count; ++worker) {
        const std::size_t begin = worker * chunk_size;
        const std::size_t end = std::min(attempts, begin + chunk_size);
        if (begin >= end) {
            break;
        }
        workers.emplace_back([&backend, &params, &passwords, &worker_timings, timings, worker, begin, end]() {
            const auto worker_start = std::chrono::steady_clock::now();
            Argon2FirstBlockTimings* local_timings = timings == nullptr ? nullptr : &worker_timings[worker];
            for (std::size_t i = begin; i < end; ++i) {
                fillPasswordBlock(backend, params, i, passwords[i], local_timings);
            }
            if (local_timings != nullptr) {
                local_timings->worker_ms = elapsedMillis(worker_start, std::chrono::steady_clock::now());
            }
        });
    }

    for (std::thread& worker : workers) {
        worker.join();
    }
    if (timings != nullptr) {
        for (const Argon2FirstBlockTimings& item : worker_timings) {
            timings->initial_hash_ms += item.initial_hash_ms;
            timings->digest_ms += item.digest_ms;
            timings->worker_ms = std::max(timings->worker_ms, item.worker_ms);
        }
    }
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
        auto timed_setup_step = [&request](auto&& action) {
            if (!request.detailed_timings) {
                action();
                return 0.0;
            }
            const auto step_start = std::chrono::steady_clock::now();
            action();
            return elapsedMillis(step_start, std::chrono::steady_clock::now());
        };
        std::string salt;
        std::string prefix;
        std::string fixed_key;
        result.timings.setup_normalize_cpu_ms = timed_setup_step([&]() {
            salt = normalizeHex(request.salt_hex);
            prefix = normalizeHex(request.key_prefix);
            fixed_key = normalizeHex(request.key);
        });
        const bool single_key = !fixed_key.empty();
        const std::size_t attempts = single_key ? 1 : request.batch_size;
        result.first_block_worker_count = firstBlockWorkerCount(attempts, request.first_block_workers);
        result.first_block_chunk_size = firstBlockChunkSize(attempts, result.first_block_worker_count);

        auto& compute_backend = backend();
        result.timings.setup_activate_cpu_ms = timed_setup_step([&]() {
            compute_backend.activate();
        });
        DeviceInfo device_info{};
        result.timings.setup_device_info_cpu_ms = timed_setup_step([&]() {
            device_info = compute_backend.getDeviceInfo();
        });
        result.device_id = device_info.index;

        Argon2Params params;
        result.timings.setup_params_cpu_ms = timed_setup_step([&]() {
            params = Argon2Params(argon2::ARGON2_ID, argon2::ARGON2_VERSION_13,
                                  kDefaultHashLength, salt, nullptr, 0, nullptr, 0,
                                  1, request.difficulty, 1);
        });
        result.timings.setup_backend_init_cpu_ms = timed_setup_step([&]() {
            ensureInitialized(compute_backend, params, attempts);
        });
        result.timings.setup_ms = elapsedMillis(setup_start, std::chrono::steady_clock::now());

        const auto input_start = std::chrono::steady_clock::now();
        Argon2FirstBlockTimings first_block_timings;
        Argon2FirstBlockTimings* detailed_first_block_timings =
            request.detailed_timings ? &first_block_timings : nullptr;
        password_storage_.clear();
        password_storage_.reserve(attempts);

        if (result.first_block_worker_count <= 1) {
            if (single_key) {
                const auto keygen_start = std::chrono::steady_clock::now();
                password_storage_.push_back(fixed_key);
                const auto keygen_end = std::chrono::steady_clock::now();
                result.timings.keygen_ms += elapsedMillis(keygen_start, keygen_end);

                const auto first_block_start = std::chrono::steady_clock::now();
                fillPasswordBlock(compute_backend, params, 0, password_storage_.front(), detailed_first_block_timings);
                const auto first_block_end = std::chrono::steady_clock::now();
                result.timings.first_block_ms += elapsedMillis(first_block_start, first_block_end);
            } else {
                RandomHexKeyGenerator key_generator(prefix, kHashApiKeyLength);
                for (std::size_t i = 0; i < attempts; ++i) {
                    const auto keygen_start = std::chrono::steady_clock::now();
                    const std::string key = key_generator.nextRandomKey();
                    const auto keygen_end = std::chrono::steady_clock::now();
                    result.timings.keygen_ms += elapsedMillis(keygen_start, keygen_end);

                    const auto first_block_start = std::chrono::steady_clock::now();
                    fillPasswordBlock(compute_backend, params, i, key, detailed_first_block_timings);
                    const auto first_block_end = std::chrono::steady_clock::now();
                    result.timings.first_block_ms += elapsedMillis(first_block_start, first_block_end);
                    password_storage_.push_back(key);
                }
            }
        } else {
            const auto keygen_start = std::chrono::steady_clock::now();
            RandomHexKeyGenerator key_generator(prefix, kHashApiKeyLength);
            for (std::size_t i = 0; i < attempts; ++i) {
                password_storage_.push_back(key_generator.nextRandomKey());
            }
            result.timings.keygen_ms = elapsedMillis(keygen_start, std::chrono::steady_clock::now());

            const auto first_block_start = std::chrono::steady_clock::now();
            fillPasswordBlocks(compute_backend, params, password_storage_, request.first_block_workers, detailed_first_block_timings);
            result.timings.first_block_ms = elapsedMillis(first_block_start, std::chrono::steady_clock::now());
        }
        result.timings.first_block_initial_hash_cpu_ms = first_block_timings.initial_hash_ms;
        result.timings.first_block_digest_cpu_ms = first_block_timings.digest_ms;
        result.timings.first_block_max_worker_ms = first_block_timings.worker_ms;
        result.timings.input_ms = elapsedMillis(input_start, std::chrono::steady_clock::now());

        const auto compute_start = std::chrono::steady_clock::now();
        compute_backend.run();
        result.timings.kernel_ms = static_cast<double>(compute_backend.finish());
        result.timings.host_to_device_ms = static_cast<double>(compute_backend.getLastHostToDeviceMs());
        result.timings.device_to_host_ms = static_cast<double>(compute_backend.getLastDeviceToHostMs());
        result.timings.compute_ms = elapsedMillis(compute_start, std::chrono::steady_clock::now());

        const auto finalize_start = std::chrono::steady_clock::now();
        std::array<std::array<std::uint8_t, kDefaultHashLength>, kFinalizeTimingChunkSize> finalized_buffers;
        std::vector<std::string> finalized_hashes;
        finalized_hashes.reserve(std::min(kFinalizeTimingChunkSize, attempts));
        for (std::size_t begin = 0; begin < attempts; begin += kFinalizeTimingChunkSize) {
            const std::size_t end = std::min(attempts, begin + kFinalizeTimingChunkSize);
            finalized_hashes.clear();
            finalized_hashes.reserve(end - begin);

            const auto finalize_hash_start = std::chrono::steady_clock::now();
            const auto argon2_finalize_start = std::chrono::steady_clock::now();
            for (std::size_t i = begin; i < end; ++i) {
                params.finalize(finalized_buffers[i - begin].data(), compute_backend.getOutputMemory(i));
            }
            result.timings.argon2_finalize_ms += elapsedMillis(
                argon2_finalize_start,
                std::chrono::steady_clock::now());

            const auto base64_start = std::chrono::steady_clock::now();
            for (std::size_t i = begin; i < end; ++i) {
                finalized_hashes.push_back(base64Encode(finalized_buffers[i - begin].data(), kDefaultHashLength));
            }
            result.timings.base64_ms += elapsedMillis(base64_start, std::chrono::steady_clock::now());
            result.timings.finalize_hash_ms += elapsedMillis(finalize_hash_start, std::chrono::steady_clock::now());

            const auto match_start = std::chrono::steady_clock::now();
            for (std::size_t offset = 0; offset < finalized_hashes.size(); ++offset) {
                const std::size_t i = begin + offset;
                const std::string& hash = finalized_hashes[offset];
                const std::string& key = password_storage_[i];
                if (single_key) {
                    result.hash = hash;
                }
                appendMatches(request, result, key, hash, i);
            }
            result.timings.match_ms += elapsedMillis(match_start, std::chrono::steady_clock::now());
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
