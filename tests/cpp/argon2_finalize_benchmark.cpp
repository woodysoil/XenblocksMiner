#include "argon2params.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kHashLength = 64;
constexpr std::size_t kBlockSize = argon2::ARGON2_BLOCK_SIZE;
constexpr std::uint64_t kFnvOffset = UINT64_C(14695981039346656037);
constexpr std::uint64_t kFnvPrime = UINT64_C(1099511628211);
constexpr const char* kExpectedDefaultSampleHash =
    "8a819c67c36ca294116d0fd0fa341940cbe08fa1e138fef94b3cc0bd0a503aeca6b6b19bb057a011"
    "ff4377903d15b0bfc0d53498d8ba08b7790d9a9345e99595";
constexpr std::uint64_t kExpectedDefaultChecksum = UINT64_C(0xa64548db96f51b25);

std::size_t parseSizeArg(int argc, char** argv, const char* name, std::size_t fallback)
{
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], name) == 0) {
            char* end = nullptr;
            const unsigned long long value = std::strtoull(argv[i + 1], &end, 10);
            if (end != argv[i + 1] && *end == '\0' && value > 0) {
                return static_cast<std::size_t>(value);
            }
        }
    }
    return fallback;
}

void fillDeterministicBlocks(std::vector<std::uint8_t>& blocks)
{
    std::uint32_t state = UINT32_C(0x6d2b79f5);
    for (std::uint8_t& byte : blocks) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        byte = static_cast<std::uint8_t>(state & UINT32_C(0xff));
    }
}

std::string hexBytes(const std::uint8_t* bytes, std::size_t size)
{
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (std::size_t i = 0; i < size; ++i) {
        out << std::setw(2) << static_cast<unsigned int>(bytes[i]);
    }
    return out.str();
}

std::uint64_t updateChecksum(std::uint64_t checksum, const std::uint8_t* bytes, std::size_t size)
{
    for (std::size_t i = 0; i < size; ++i) {
        checksum ^= bytes[i];
        checksum *= kFnvPrime;
    }
    return checksum;
}

double elapsedMillis(std::chrono::steady_clock::time_point start,
                     std::chrono::steady_clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

} // namespace

int main(int argc, char** argv)
{
    const std::size_t block_count = parseSizeArg(argc, argv, "--blocks", 4096);
    const std::size_t repeat = parseSizeArg(argc, argv, "--repeat", 128);
    const std::size_t iterations = block_count * repeat;

    std::vector<std::uint8_t> blocks(block_count * kBlockSize);
    fillDeterministicBlocks(blocks);

    Argon2Params params(argon2::ARGON2_ID,
                        argon2::ARGON2_VERSION_13,
                        kHashLength,
                        "aabbccddeeff0011",
                        nullptr,
                        0,
                        nullptr,
                        0,
                        1,
                        8,
                        1);

    std::vector<std::uint8_t> first_hash(kHashLength);
    std::vector<std::uint8_t> check_hash(kHashLength);
    std::vector<std::uint8_t> current_hash(kHashLength);
    params.finalize(first_hash.data(), blocks.data());

    std::uint64_t checksum = kFnvOffset;
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t r = 0; r < repeat; ++r) {
        for (std::size_t block = 0; block < block_count; ++block) {
            const std::uint8_t* memory = blocks.data() + block * kBlockSize;
            params.finalize(current_hash.data(), memory);
            checksum = updateChecksum(checksum, current_hash.data(), current_hash.size());
        }
    }
    const double elapsed_ms = elapsedMillis(start, std::chrono::steady_clock::now());

    params.finalize(check_hash.data(), blocks.data());
    const std::string sample_hash = hexBytes(first_hash.data(), first_hash.size());
    const bool default_shape = block_count == 4096 && repeat == 128;
    const bool deterministic = first_hash == check_hash;
    const bool known_sample_ok = !default_shape || sample_hash == kExpectedDefaultSampleHash;
    const bool known_checksum_ok = !default_shape || checksum == kExpectedDefaultChecksum;
    const bool ok = deterministic && known_sample_ok && known_checksum_ok;
    const double finalizes_per_second = elapsed_ms > 0.0
        ? static_cast<double>(iterations) / (elapsed_ms / 1000.0)
        : 0.0;
    const double ns_per_finalize = iterations > 0
        ? elapsed_ms * 1000000.0 / static_cast<double>(iterations)
        : 0.0;

    std::cout
        << "{"
        << "\"ok\":" << (ok ? "true" : "false") << ","
        << "\"algorithm\":\"argon2id-xen-finalize\","
        << "\"block_size\":" << kBlockSize << ","
        << "\"blocks\":" << block_count << ","
        << "\"repeat\":" << repeat << ","
        << "\"iterations\":" << iterations << ","
        << "\"elapsed_ms\":" << elapsed_ms << ","
        << "\"finalizes_per_second\":" << finalizes_per_second << ","
        << "\"ns_per_finalize\":" << ns_per_finalize << ","
        << "\"checksum\":\"" << std::hex << checksum << std::dec << "\","
        << "\"sample_hash\":\"" << sample_hash << "\","
        << "\"deterministic\":" << (deterministic ? "true" : "false") << ","
        << "\"known_sample_ok\":" << (known_sample_ok ? "true" : "false") << ","
        << "\"known_checksum_ok\":" << (known_checksum_ok ? "true" : "false")
        << "}\n";

    return ok ? 0 : 1;
}
