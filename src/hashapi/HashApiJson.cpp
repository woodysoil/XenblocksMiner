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
        << "\"input_ms\":" << timings.input_ms << ","
        << "\"keygen_ms\":" << timings.keygen_ms << ","
        << "\"first_block_ms\":" << timings.first_block_ms << ","
        << "\"compute_ms\":" << timings.compute_ms << ","
        << "\"kernel_ms\":" << timings.kernel_ms << ","
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
        << "\"attempts\":" << result.attempts << ","
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
