#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace hashapi {

constexpr std::size_t kHashApiKeyLength = 64;
constexpr std::size_t kDefaultHashLength = 64;
constexpr std::size_t kMaxTargetPatternLength = 128;
constexpr std::size_t kMaxCpuBatchSize = 10000;

struct HashApiRequest {
    std::string request_id;
    std::string algorithm = "argon2id-xen";
    std::string backend = "cpu";
    std::string salt_hex;
    std::string key;
    std::string key_prefix;
    std::string target_pattern = "XEN11";
    std::uint32_t difficulty = 42069;
    std::size_t batch_size = 1;
    int device_id = 0;
    bool allow_xuni = true;
};

struct HashApiMatch {
    std::string key;
    std::string hash;
    std::string matched_pattern;
    std::size_t attempt_index = 0;
    bool is_superblock = false;
};

struct HashApiTimings {
    double validation_ms = 0.0;
    double setup_ms = 0.0;
    double input_ms = 0.0;
    double keygen_ms = 0.0;
    double first_block_ms = 0.0;
    double compute_ms = 0.0;
    double kernel_ms = 0.0;
    double finalize_ms = 0.0;
    double finalize_hash_ms = 0.0;
    double match_ms = 0.0;
    double total_ms = 0.0;
};

struct HashApiResult {
    std::string request_id;
    bool ok = false;
    std::string error;
    std::string algorithm;
    std::string backend;
    int device_id = 0;
    std::size_t batch_size = 0;
    std::size_t attempts = 0;
    double elapsed_ms = 0.0;
    double hashrate = 0.0;
    HashApiTimings timings;
    std::string hash;
    std::vector<HashApiMatch> matches;
};

class IHashBackend {
public:
    virtual ~IHashBackend() = default;
    virtual HashApiResult runBatch(const HashApiRequest& request) = 0;
};

} // namespace hashapi
