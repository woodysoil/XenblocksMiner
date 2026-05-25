#include "HashApiJson.h"

#include <iomanip>
#include <sstream>

namespace hashapi {
namespace {

std::string escapeJson(const std::string& value)
{
    std::ostringstream out;
    for (unsigned char ch : value) {
        switch (ch) {
        case '\\': out << "\\\\"; break;
        case '"': out << "\\\""; break;
        case '\b': out << "\\b"; break;
        case '\f': out << "\\f"; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default:
            if (ch < 0x20) {
                out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                    << static_cast<int>(ch);
            } else {
                out << ch;
            }
        }
    }
    return out.str();
}

std::string quote(const std::string& value)
{
    return "\"" + escapeJson(value) + "\"";
}

const char* boolText(bool value)
{
    return value ? "true" : "false";
}

} // namespace

std::string toJson(const HashApiMatch& match)
{
    std::ostringstream out;
    out << "{"
        << "\"key\":" << quote(match.key) << ","
        << "\"hash\":" << quote(match.hash) << ","
        << "\"matched_pattern\":" << quote(match.matched_pattern) << ","
        << "\"attempt_index\":" << match.attempt_index << ","
        << "\"is_superblock\":" << boolText(match.is_superblock)
        << "}";
    return out.str();
}

std::string toJson(const HashApiTimings& timings)
{
    std::ostringstream out;
    out << "{"
        << "\"validation_ms\":" << timings.validation_ms << ","
        << "\"setup_ms\":" << timings.setup_ms << ","
        << "\"setup_normalize_cpu_ms\":" << timings.setup_normalize_cpu_ms << ","
        << "\"setup_activate_cpu_ms\":" << timings.setup_activate_cpu_ms << ","
        << "\"setup_device_info_cpu_ms\":" << timings.setup_device_info_cpu_ms << ","
        << "\"setup_params_cpu_ms\":" << timings.setup_params_cpu_ms << ","
        << "\"setup_backend_init_cpu_ms\":" << timings.setup_backend_init_cpu_ms << ","
        << "\"input_ms\":" << timings.input_ms << ","
        << "\"keygen_ms\":" << timings.keygen_ms << ","
        << "\"first_block_ms\":" << timings.first_block_ms << ","
        << "\"first_block_initial_hash_cpu_ms\":" << timings.first_block_initial_hash_cpu_ms << ","
        << "\"first_block_digest_cpu_ms\":" << timings.first_block_digest_cpu_ms << ","
        << "\"first_block_max_worker_ms\":" << timings.first_block_max_worker_ms << ","
        << "\"first_block_thread_launch_ms\":" << timings.first_block_thread_launch_ms << ","
        << "\"first_block_max_worker_start_ms\":" << timings.first_block_max_worker_start_ms << ","
        << "\"first_block_worker_start_span_ms\":" << timings.first_block_worker_start_span_ms << ","
        << "\"first_block_max_worker_finish_ms\":" << timings.first_block_max_worker_finish_ms << ","
        << "\"first_block_worker_finish_span_ms\":" << timings.first_block_worker_finish_span_ms << ","
        << "\"compute_ms\":" << timings.compute_ms << ","
        << "\"kernel_ms\":" << timings.kernel_ms << ","
        << "\"host_to_device_ms\":" << timings.host_to_device_ms << ","
        << "\"gpu_first_block_ms\":" << timings.gpu_first_block_ms << ","
        << "\"device_to_host_ms\":" << timings.device_to_host_ms << ","
        << "\"finalize_ms\":" << timings.finalize_ms << ","
        << "\"finalize_hash_ms\":" << timings.finalize_hash_ms << ","
        << "\"argon2_finalize_ms\":" << timings.argon2_finalize_ms << ","
        << "\"base64_ms\":" << timings.base64_ms << ","
        << "\"match_ms\":" << timings.match_ms << ","
        << "\"total_ms\":" << timings.total_ms
        << "}";
    return out.str();
}

std::string toJson(const HashApiResult& result)
{
    std::ostringstream out;
    out << "{"
        << "\"request_id\":" << quote(result.request_id) << ","
        << "\"ok\":" << boolText(result.ok) << ","
        << "\"error\":" << quote(result.error) << ","
        << "\"algorithm\":" << quote(result.algorithm) << ","
        << "\"backend\":" << quote(result.backend) << ","
        << "\"device_id\":" << result.device_id << ","
        << "\"batch_size\":" << result.batch_size << ","
        << "\"batch_size_min\":" << result.batch_size_min << ","
        << "\"batch_size_max\":" << result.batch_size_max << ","
        << "\"attempts\":" << result.attempts << ","
        << "\"first_block_dynamic_chunk_size\":" << result.first_block_dynamic_chunk_size << ","
        << "\"first_block_dynamic_chunk_auto\":" << boolText(result.first_block_dynamic_chunk_auto) << ","
        << "\"first_block_worker_count\":" << result.first_block_worker_count << ","
        << "\"first_block_chunk_size\":" << result.first_block_chunk_size << ","
        << "\"first_block_dynamic_chunk_size_min\":" << result.first_block_dynamic_chunk_size_min << ","
        << "\"first_block_dynamic_chunk_size_max\":" << result.first_block_dynamic_chunk_size_max << ","
        << "\"first_block_chunk_size_min\":" << result.first_block_chunk_size_min << ","
        << "\"first_block_chunk_size_max\":" << result.first_block_chunk_size_max << ","
        << "\"gpu_first_blocks\":" << boolText(result.gpu_first_blocks) << ","
        << "\"elapsed_ms\":" << result.elapsed_ms << ","
        << "\"hashrate\":" << result.hashrate << ","
        << "\"timings\":" << toJson(result.timings) << ","
        << "\"hash\":" << quote(result.hash) << ","
        << "\"matches\":[";

    for (std::size_t i = 0; i < result.matches.size(); ++i) {
        if (i > 0) {
            out << ",";
        }
        out << toJson(result.matches[i]);
    }

    out << "]}";
    return out.str();
}

} // namespace hashapi
