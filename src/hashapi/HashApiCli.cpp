#include "HashApiCli.h"

#include "CpuHashBackend.h"
#include "HashApiJson.h"
#include "HashApiTuning.h"
#include "HashApiValidation.h"
#if defined(XENBLOCKS_BUILD_MINER)
#include "CudaHashBackend.h"
#include "../CudaBackend.h"
#include "../CudaException.h"
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <chrono>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace hashapi {
namespace {

void printUsage()
{
    std::cout
        << "Hash API commands:\n"
        << "  xenblocksMiner hash-one --salt <hex> --key <64-hex> [--backend cpu|cuda] [--difficulty <n>] [--no-xuni] [--detailed-timings] [--first-block-workers <n>] [--first-block-dynamic-chunk-size <n>] [--first-block-dynamic-chunk-auto] [--gpu-first-blocks] [--json]\n"
        << "  xenblocksMiner hash-batch --salt <hex> [--backend cpu|cuda] [--prefix <hex>] [--pattern XEN11] [--batch-size <n>] [--auto-batch-size] [--difficulty <n>] [--no-xuni] [--detailed-timings] [--first-block-workers <n>] [--first-block-dynamic-chunk-size <n>] [--first-block-dynamic-chunk-auto] [--gpu-first-blocks] [--json]\n"
        << "  xenblocksMiner hash-benchmark --salt <hex> [--backend cpu|cuda] [--key <64-hex>] [--prefix <hex>] [--seconds <n>] [--batch-size <n>] [--auto-batch-size] [--batch-size-sequence <n,n,...>] [--difficulty <n>] [--difficulty-sequence <n,n,...>] [--no-xuni] [--detailed-timings] [--first-block-workers <n>] [--first-block-dynamic-chunk-size <n>] [--first-block-dynamic-chunk-auto] [--gpu-first-blocks] [--json]\n";
}

std::unordered_map<std::string, std::string> parseArgs(int argc, const char* const* argv)
{
    std::unordered_map<std::string, std::string> args;
    for (int i = 2; i < argc; ++i) {
        std::string key = argv[i];
        if (key.rfind("--", 0) != 0) {
            continue;
        }
        if (key == "--json" || key == "--no-xuni" || key == "--detailed-timings" ||
            key == "--auto-batch-size" ||
            key == "--first-block-dynamic-chunk-auto" ||
            key == "--gpu-first-blocks") {
            args[key] = "true";
            continue;
        }
        if (i + 1 < argc) {
            args[key] = argv[++i];
        }
    }
    return args;
}

bool hasArg(const std::unordered_map<std::string, std::string>& args,
            const std::string& key)
{
    return args.find(key) != args.end();
}

std::string getArg(const std::unordered_map<std::string, std::string>& args,
                   const std::string& key,
                   const std::string& fallback = "")
{
    auto it = args.find(key);
    return it == args.end() ? fallback : it->second;
}

std::uint32_t getUIntArg(const std::unordered_map<std::string, std::string>& args,
                         const std::string& key,
                         std::uint32_t fallback)
{
    auto it = args.find(key);
    if (it == args.end()) {
        return fallback;
    }
    return static_cast<std::uint32_t>(std::stoul(it->second));
}

std::size_t getSizeArg(const std::unordered_map<std::string, std::string>& args,
                       const std::string& key,
                       std::size_t fallback)
{
    auto it = args.find(key);
    if (it == args.end()) {
        return fallback;
    }
    return static_cast<std::size_t>(std::stoull(it->second));
}

std::vector<std::uint32_t> parseDifficultySequence(const std::string& text)
{
    std::vector<std::uint32_t> values;
    if (text.empty()) {
        return values;
    }

    std::size_t start = 0;
    while (start <= text.size()) {
        const std::size_t end = text.find(',', start);
        const std::string token = text.substr(start, end == std::string::npos ? std::string::npos : end - start);
        if (token.empty()) {
            throw std::runtime_error("difficulty sequence cannot contain empty values");
        }
        for (char ch : token) {
            if (ch < '0' || ch > '9') {
                throw std::runtime_error("difficulty sequence values must be unsigned integers");
            }
        }

        std::size_t parsed = 0;
        const unsigned long value = std::stoul(token, &parsed);
        if (parsed != token.size()) {
            throw std::runtime_error("difficulty sequence values must be unsigned integers");
        }
        if (value == 0 || value > std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error("difficulty sequence values must be between 1 and UINT32_MAX");
        }
        values.push_back(static_cast<std::uint32_t>(value));

        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }

    return values;
}

std::vector<std::size_t> parseBatchSizeSequence(const std::string& text)
{
    std::vector<std::size_t> values;
    if (text.empty()) {
        return values;
    }

    std::size_t start = 0;
    while (start <= text.size()) {
        const std::size_t end = text.find(',', start);
        const std::string token = text.substr(start, end == std::string::npos ? std::string::npos : end - start);
        if (token.empty()) {
            throw std::runtime_error("batch-size sequence cannot contain empty values");
        }
        for (char ch : token) {
            if (ch < '0' || ch > '9') {
                throw std::runtime_error("batch-size sequence values must be unsigned integers");
            }
        }

        std::size_t parsed = 0;
        const unsigned long long value = std::stoull(token, &parsed);
        if (parsed != token.size()) {
            throw std::runtime_error("batch-size sequence values must be unsigned integers");
        }
        if (value == 0 || value > static_cast<unsigned long long>(std::numeric_limits<std::size_t>::max())) {
            throw std::runtime_error("batch-size sequence values must be between 1 and SIZE_MAX");
        }
        values.push_back(static_cast<std::size_t>(value));

        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }

    return values;
}

void addTimings(HashApiTimings& target, const HashApiTimings& source)
{
    target.validation_ms += source.validation_ms;
    target.setup_ms += source.setup_ms;
    target.setup_normalize_cpu_ms += source.setup_normalize_cpu_ms;
    target.setup_activate_cpu_ms += source.setup_activate_cpu_ms;
    target.setup_device_info_cpu_ms += source.setup_device_info_cpu_ms;
    target.setup_params_cpu_ms += source.setup_params_cpu_ms;
    target.setup_backend_init_cpu_ms += source.setup_backend_init_cpu_ms;
    target.input_ms += source.input_ms;
    target.keygen_ms += source.keygen_ms;
    target.first_block_ms += source.first_block_ms;
    target.first_block_initial_hash_cpu_ms += source.first_block_initial_hash_cpu_ms;
    target.first_block_digest_cpu_ms += source.first_block_digest_cpu_ms;
    target.first_block_max_worker_ms += source.first_block_max_worker_ms;
    target.first_block_thread_launch_ms += source.first_block_thread_launch_ms;
    target.first_block_max_worker_start_ms += source.first_block_max_worker_start_ms;
    target.first_block_worker_start_span_ms += source.first_block_worker_start_span_ms;
    target.first_block_max_worker_finish_ms += source.first_block_max_worker_finish_ms;
    target.first_block_worker_finish_span_ms += source.first_block_worker_finish_span_ms;
    target.compute_ms += source.compute_ms;
    target.kernel_ms += source.kernel_ms;
    target.host_to_device_ms += source.host_to_device_ms;
    target.gpu_first_block_ms += source.gpu_first_block_ms;
    target.device_to_host_ms += source.device_to_host_ms;
    target.finalize_ms += source.finalize_ms;
    target.finalize_hash_ms += source.finalize_hash_ms;
    target.argon2_finalize_ms += source.argon2_finalize_ms;
    target.base64_ms += source.base64_ms;
    target.match_ms += source.match_ms;
    target.total_ms += source.total_ms;
}

HashApiRequest baseRequest(const std::unordered_map<std::string, std::string>& args)
{
    HashApiRequest request;
    request.request_id = getArg(args, "--request-id", "");
    request.backend = getArg(args, "--backend", "cpu");
    request.salt_hex = getArg(args, "--salt");
    request.key = getArg(args, "--key");
    request.key_prefix = getArg(args, "--prefix");
    request.target_pattern = getArg(args, "--pattern", "XEN11");
    request.difficulty = getUIntArg(args, "--difficulty", request.difficulty);
    request.batch_size = getSizeArg(args, "--batch-size", request.batch_size);
    request.device_id = static_cast<int>(getUIntArg(args, "--device", 0));
    request.allow_xuni = getArg(args, "--no-xuni") != "true";
    request.detailed_timings = getArg(args, "--detailed-timings") == "true";
    request.first_block_workers = getSizeArg(args, "--first-block-workers", 0);
    request.first_block_dynamic_chunk_size = getSizeArg(args, "--first-block-dynamic-chunk-size", 0);
    request.first_block_dynamic_chunk_auto = getArg(args, "--first-block-dynamic-chunk-auto") == "true";
    request.gpu_first_blocks = getArg(args, "--gpu-first-blocks") == "true";
    return request;
}

int printResult(const HashApiResult& result, bool json)
{
    if (json) {
        std::cout << toJson(result) << std::endl;
    } else if (!result.ok) {
        std::cerr << "Hash API error: " << result.error << std::endl;
    } else {
        std::cout << "ok=true"
                  << " backend=" << result.backend
                  << " attempts=" << result.attempts
                  << " hashrate=" << result.hashrate
                  << " matches=" << result.matches.size()
                  << std::endl;
        if (!result.hash.empty()) {
            std::cout << "hash=" << result.hash << std::endl;
        }
    }
    return result.ok ? 0 : 2;
}

HashApiResult runBackend(const HashApiRequest& request)
{
    if (request.backend == "cuda") {
        const auto errors = validateRequest(request);
        if (!errors.empty()) {
            HashApiResult result;
            result.request_id = request.request_id;
            result.algorithm = request.algorithm;
            result.backend = request.backend;
            result.device_id = request.device_id;
            result.batch_size = request.batch_size;
            result.error = joinErrors(errors);
            return result;
        }
#if defined(XENBLOCKS_BUILD_MINER)
        try {
            CudaHashBackend backend(std::make_unique<CudaBackend>(request.device_id));
            return backend.runBatch(request);
        } catch (const std::exception& ex) {
            HashApiResult result;
            result.request_id = request.request_id;
            result.algorithm = request.algorithm;
            result.backend = "cuda";
            result.device_id = request.device_id;
            result.batch_size = request.batch_size;
            result.error = ex.what();
            return result;
        }
#else
        HashApiResult result;
        result.request_id = request.request_id;
        result.algorithm = request.algorithm;
        result.backend = "cuda";
        result.device_id = request.device_id;
        result.batch_size = request.batch_size;
        result.error = "cuda backend is not available in this build";
        return result;
#endif
    }

    CpuHashBackend backend;
    return backend.runBatch(request);
}

std::unique_ptr<IHashBackend> makeReusableBackend(const HashApiRequest& request)
{
    if (request.backend == "cuda") {
#if defined(XENBLOCKS_BUILD_MINER)
        return std::make_unique<CudaHashBackend>(std::make_unique<CudaBackend>(request.device_id));
#else
        throw std::runtime_error("cuda backend is not available in this build");
#endif
    }

    return std::make_unique<CpuHashBackend>();
}

#if defined(XENBLOCKS_BUILD_MINER)
std::size_t queryCudaFreeMemory(int device_id)
{
    int previous_device = 0;
    const cudaError_t current_result = cudaGetDevice(&previous_device);
    const bool has_previous_device = current_result == cudaSuccess;

    CudaException::check(cudaSetDevice(device_id));
    std::size_t free_memory = 0;
    std::size_t total_memory = 0;
    CudaException::check(cudaMemGetInfo(&free_memory, &total_memory));

    if (has_previous_device && previous_device != device_id) {
        CudaException::check(cudaSetDevice(previous_device));
    }
    return free_memory;
}
#endif

std::size_t selectAutomaticCudaBatchSize(const HashApiRequest& request,
                                         const std::vector<std::uint32_t>& difficulty_sequence,
                                         std::size_t explicit_max_batch_size)
{
    if (request.backend != "cuda") {
        throw std::runtime_error("--auto-batch-size is only supported with --backend cuda");
    }
#if defined(XENBLOCKS_BUILD_MINER)
    const std::size_t free_memory = queryCudaFreeMemory(request.device_id);
    const auto decision = difficulty_sequence.empty()
        ? selectCudaBatchSize(free_memory, request.difficulty, explicit_max_batch_size)
        : selectCudaBatchSizeForDifficultySequence(free_memory, difficulty_sequence, explicit_max_batch_size);
    if (decision.selected_batch_size == 0) {
        throw std::runtime_error("automatic CUDA batch-size selection found no safe batch size");
    }
    return decision.selected_batch_size;
#else
    throw std::runtime_error("automatic CUDA batch-size selection is not available in this build");
#endif
}

std::vector<std::uint32_t> benchmarkDifficulties(const HashApiRequest& request,
                                                 const std::vector<std::uint32_t>& difficulty_sequence)
{
    if (!difficulty_sequence.empty()) {
        return difficulty_sequence;
    }
    return {request.difficulty};
}

std::vector<std::size_t> benchmarkBatchSizes(const HashApiRequest& request,
                                             const std::vector<std::size_t>& batch_size_sequence)
{
    if (!batch_size_sequence.empty()) {
        return batch_size_sequence;
    }
    return {request.batch_size};
}

int runBenchmark(HashApiRequest request,
                 std::uint32_t seconds,
                 bool json,
                 const std::vector<std::uint32_t>& difficulty_sequence,
                 const std::vector<std::size_t>& batch_size_sequence)
{
    const auto difficulties = benchmarkDifficulties(request, difficulty_sequence);
    const auto batch_sizes = benchmarkBatchSizes(request, batch_size_sequence);
    if (difficulties.size() != batch_sizes.size() &&
        difficulties.size() != 1 &&
        batch_sizes.size() != 1) {
        HashApiResult result;
        result.request_id = request.request_id;
        result.algorithm = request.algorithm;
        result.backend = request.backend;
        result.device_id = request.device_id;
        result.batch_size = request.batch_size;
        result.error = "difficulty sequence and batch-size sequence lengths must match unless one sequence has length 1";
        return printResult(result, json);
    }

    const std::size_t shape_count = std::max(difficulties.size(), batch_sizes.size());
    for (std::size_t i = 0; i < shape_count; ++i) {
        HashApiRequest validation_request = request;
        validation_request.difficulty = difficulties[difficulties.size() == 1 ? 0 : i];
        validation_request.batch_size = batch_sizes[batch_sizes.size() == 1 ? 0 : i];
        const auto errors = validateRequest(validation_request);
        if (!errors.empty()) {
            HashApiResult result;
            result.request_id = request.request_id;
            result.algorithm = request.algorithm;
            result.backend = request.backend;
            result.device_id = request.device_id;
            result.batch_size = validation_request.batch_size;
            result.error = difficulty_sequence.empty() && batch_size_sequence.empty()
                ? joinErrors(errors)
                : "benchmark sequence item " + std::to_string(i) + ": " + joinErrors(errors);
            return printResult(result, json);
        }
    }

    std::unique_ptr<IHashBackend> backend;
    try {
        backend = makeReusableBackend(request);
    } catch (const std::exception& ex) {
        HashApiResult result;
        result.request_id = request.request_id;
        result.algorithm = request.algorithm;
        result.backend = request.backend;
        result.device_id = request.device_id;
        result.batch_size = request.batch_size;
        result.error = ex.what();
        return printResult(result, json);
    }

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
    std::size_t difficulty_index = 0;
    HashApiResult aggregate;
    aggregate.request_id = request.request_id;
    aggregate.algorithm = request.algorithm;
    aggregate.backend = request.backend;
    aggregate.device_id = request.device_id;
    aggregate.batch_size = request.batch_size;
    bool batch_size_range_seen = false;
    auto update_batch_size_ranges = [&aggregate, &batch_size_range_seen](std::size_t batch_size) {
        if (!batch_size_range_seen) {
            aggregate.batch_size_min = batch_size;
            aggregate.batch_size_max = batch_size;
            batch_size_range_seen = true;
            return;
        }
        aggregate.batch_size_min = std::min(aggregate.batch_size_min, batch_size);
        aggregate.batch_size_max = std::max(aggregate.batch_size_max, batch_size);
    };
    bool first_block_range_seen = false;
    auto update_first_block_ranges = [&aggregate, &first_block_range_seen](const HashApiResult& current) {
        if (!first_block_range_seen) {
            aggregate.first_block_dynamic_chunk_size_min = current.first_block_dynamic_chunk_size;
            aggregate.first_block_dynamic_chunk_size_max = current.first_block_dynamic_chunk_size;
            aggregate.first_block_chunk_size_min = current.first_block_chunk_size;
            aggregate.first_block_chunk_size_max = current.first_block_chunk_size;
            first_block_range_seen = true;
            return;
        }
        aggregate.first_block_dynamic_chunk_size_min = std::min(
            aggregate.first_block_dynamic_chunk_size_min,
            current.first_block_dynamic_chunk_size);
        aggregate.first_block_dynamic_chunk_size_max = std::max(
            aggregate.first_block_dynamic_chunk_size_max,
            current.first_block_dynamic_chunk_size);
        aggregate.first_block_chunk_size_min = std::min(
            aggregate.first_block_chunk_size_min,
            current.first_block_chunk_size);
        aggregate.first_block_chunk_size_max = std::max(
            aggregate.first_block_chunk_size_max,
            current.first_block_chunk_size);
    };

    const auto start = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() < deadline) {
        request.difficulty = difficulties[difficulties.size() == 1 ? 0 : difficulty_index];
        request.batch_size = batch_sizes[batch_sizes.size() == 1 ? 0 : difficulty_index];
        difficulty_index = (difficulty_index + 1) % shape_count;
        HashApiResult current = backend->runBatch(request);
        if (!current.ok) {
            return printResult(current, json);
        }
        aggregate.ok = true;
        aggregate.attempts += current.attempts;
        aggregate.batch_size = current.batch_size;
        update_batch_size_ranges(current.batch_size);
        aggregate.first_block_dynamic_chunk_size = current.first_block_dynamic_chunk_size;
        aggregate.first_block_dynamic_chunk_auto = current.first_block_dynamic_chunk_auto;
        aggregate.first_block_worker_count = current.first_block_worker_count;
        aggregate.first_block_chunk_size = current.first_block_chunk_size;
        aggregate.gpu_first_blocks = current.gpu_first_blocks;
        update_first_block_ranges(current);
        addTimings(aggregate.timings, current.timings);
        if (!request.key.empty()) {
            aggregate.hash = current.hash;
        }
        aggregate.matches.insert(aggregate.matches.end(), current.matches.begin(), current.matches.end());
    }

    if (request.key.empty()) {
        aggregate.hash.clear();
    }

    const auto end = std::chrono::steady_clock::now();
    aggregate.elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    if (aggregate.elapsed_ms > 0.0) {
        aggregate.hashrate = static_cast<double>(aggregate.attempts) / (aggregate.elapsed_ms / 1000.0);
    }
    return printResult(aggregate, json);
}

} // namespace

bool isHashApiCommand(int argc, const char* const* argv)
{
    if (argc < 2) {
        return false;
    }
    const std::string command = argv[1];
    return command == "hash-one" || command == "hash-batch" || command == "hash-benchmark" ||
           command == "hash-help";
}

int runHashApiCli(int argc, const char* const* argv)
{
    if (argc < 2 || std::string(argv[1]) == "hash-help") {
        printUsage();
        return 0;
    }

    const std::string command = argv[1];
    const auto args = parseArgs(argc, argv);
    const bool json = getArg(args, "--json") == "true";

    try {
        HashApiRequest request = baseRequest(args);
        if (command == "hash-one") {
            request.key = getArg(args, "--key");
            request.batch_size = 1;
            return printResult(runBackend(request), json);
        }
        if (command == "hash-batch") {
            request.batch_size = getSizeArg(args, "--batch-size", 1);
            if (hasArg(args, "--auto-batch-size")) {
                request.batch_size = selectAutomaticCudaBatchSize(
                    request,
                    {},
                    hasArg(args, "--batch-size") ? request.batch_size : 0);
            }
            return printResult(runBackend(request), json);
        }
        if (command == "hash-benchmark") {
            request.batch_size = getSizeArg(args, "--batch-size", 1);
            const auto seconds = getUIntArg(args, "--seconds", 30);
            const auto difficulty_sequence = parseDifficultySequence(getArg(args, "--difficulty-sequence"));
            const auto batch_size_sequence = parseBatchSizeSequence(getArg(args, "--batch-size-sequence"));
            if (hasArg(args, "--auto-batch-size") && batch_size_sequence.empty()) {
                request.batch_size = selectAutomaticCudaBatchSize(
                    request,
                    difficulty_sequence,
                    hasArg(args, "--batch-size") ? request.batch_size : 0);
            }
            return runBenchmark(request, seconds, json, difficulty_sequence, batch_size_sequence);
        }
    } catch (const std::exception& ex) {
        if (json) {
            HashApiResult result;
            result.error = ex.what();
            std::cout << toJson(result) << std::endl;
        } else {
            std::cerr << "Hash API error: " << ex.what() << std::endl;
        }
        return 2;
    }

    printUsage();
    return 1;
}

} // namespace hashapi
