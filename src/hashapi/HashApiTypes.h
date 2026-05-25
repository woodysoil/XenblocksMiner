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
    bool detailed_timings = false;
    std::size_t first_block_workers = 0;
    std::size_t first_block_dynamic_chunk_size = 0;
    bool first_block_dynamic_chunk_auto = false;
    bool gpu_first_blocks = false;
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
    double setup_normalize_cpu_ms = 0.0;
    double setup_activate_cpu_ms = 0.0;
    double setup_device_info_cpu_ms = 0.0;
    double setup_params_cpu_ms = 0.0;
    double setup_backend_init_cpu_ms = 0.0;
    double input_ms = 0.0;
    double keygen_ms = 0.0;
    double first_block_ms = 0.0;
    double first_block_initial_hash_cpu_ms = 0.0;
    double first_block_digest_cpu_ms = 0.0;
    double first_block_max_worker_ms = 0.0;
    double first_block_thread_launch_ms = 0.0;
    double first_block_max_worker_start_ms = 0.0;
    double first_block_worker_start_span_ms = 0.0;
    double first_block_max_worker_finish_ms = 0.0;
    double first_block_worker_finish_span_ms = 0.0;
    double compute_ms = 0.0;
    double kernel_ms = 0.0;
    double host_to_device_ms = 0.0;
    double gpu_first_block_ms = 0.0;
    double device_to_host_ms = 0.0;
    double finalize_ms = 0.0;
    double finalize_hash_ms = 0.0;
    double argon2_finalize_ms = 0.0;
    double base64_ms = 0.0;
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
    std::size_t batch_size_min = 0;
    std::size_t batch_size_max = 0;
    std::size_t attempts = 0;
    std::size_t first_block_dynamic_chunk_size = 0;
    bool first_block_dynamic_chunk_auto = false;
    std::size_t first_block_worker_count = 0;
    std::size_t first_block_chunk_size = 0;
    std::size_t first_block_dynamic_chunk_size_min = 0;
    std::size_t first_block_dynamic_chunk_size_max = 0;
    std::size_t first_block_chunk_size_min = 0;
    std::size_t first_block_chunk_size_max = 0;
    bool gpu_first_blocks = false;
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
