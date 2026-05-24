#include "HashApiCli.h"

#include "CpuHashBackend.h"
#include "HashApiJson.h"
#include "HashApiValidation.h"
#if defined(XENBLOCKS_BUILD_MINER)
#include "CudaHashBackend.h"
#include "../CudaBackend.h"
#endif

#include <chrono>
#include <exception>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>

namespace hashapi {
namespace {

void printUsage()
{
    std::cout
        << "Hash API commands:\n"
        << "  xenblocksMiner hash-one --salt <hex> --key <64-hex> [--backend cpu|cuda] [--difficulty <n>] [--no-xuni] [--json]\n"
        << "  xenblocksMiner hash-batch --salt <hex> [--backend cpu|cuda] [--prefix <hex>] [--pattern XEN11] [--batch-size <n>] [--difficulty <n>] [--no-xuni] [--json]\n"
        << "  xenblocksMiner hash-benchmark --salt <hex> [--backend cpu|cuda] [--prefix <hex>] [--seconds <n>] [--batch-size <n>] [--difficulty <n>] [--no-xuni] [--json]\n";
}

std::unordered_map<std::string, std::string> parseArgs(int argc, const char* const* argv)
{
    std::unordered_map<std::string, std::string> args;
    for (int i = 2; i < argc; ++i) {
        std::string key = argv[i];
        if (key.rfind("--", 0) != 0) {
            continue;
        }
        if (key == "--json" || key == "--no-xuni") {
            args[key] = "true";
            continue;
        }
        if (i + 1 < argc) {
            args[key] = argv[++i];
        }
    }
    return args;
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

void addTimings(HashApiTimings& target, const HashApiTimings& source)
{
    target.validation_ms += source.validation_ms;
    target.setup_ms += source.setup_ms;
    target.input_ms += source.input_ms;
    target.compute_ms += source.compute_ms;
    target.finalize_ms += source.finalize_ms;
    target.total_ms += source.total_ms;
}

HashApiRequest baseRequest(const std::unordered_map<std::string, std::string>& args)
{
    HashApiRequest request;
    request.request_id = getArg(args, "--request-id", "");
    request.backend = getArg(args, "--backend", "cpu");
    request.salt_hex = getArg(args, "--salt");
    request.key_prefix = getArg(args, "--prefix");
    request.target_pattern = getArg(args, "--pattern", "XEN11");
    request.difficulty = getUIntArg(args, "--difficulty", request.difficulty);
    request.batch_size = getSizeArg(args, "--batch-size", request.batch_size);
    request.device_id = static_cast<int>(getUIntArg(args, "--device", 0));
    request.allow_xuni = getArg(args, "--no-xuni") != "true";
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

int runBenchmark(HashApiRequest request, std::uint32_t seconds, bool json)
{
    const auto errors = validateRequest(request);
    if (!errors.empty()) {
        HashApiResult result;
        result.request_id = request.request_id;
        result.algorithm = request.algorithm;
        result.backend = request.backend;
        result.device_id = request.device_id;
        result.batch_size = request.batch_size;
        result.error = joinErrors(errors);
        return printResult(result, json);
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
    HashApiResult aggregate;
    aggregate.request_id = request.request_id;
    aggregate.algorithm = request.algorithm;
    aggregate.backend = request.backend;
    aggregate.device_id = request.device_id;
    aggregate.batch_size = request.batch_size;

    const auto start = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() < deadline) {
        HashApiResult current = backend->runBatch(request);
        if (!current.ok) {
            return printResult(current, json);
        }
        aggregate.ok = true;
        aggregate.attempts += current.attempts;
        addTimings(aggregate.timings, current.timings);
        aggregate.matches.insert(aggregate.matches.end(), current.matches.begin(), current.matches.end());
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
            return printResult(runBackend(request), json);
        }
        if (command == "hash-benchmark") {
            request.batch_size = getSizeArg(args, "--batch-size", 1);
            const auto seconds = getUIntArg(args, "--seconds", 30);
            return runBenchmark(request, seconds, json);
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
