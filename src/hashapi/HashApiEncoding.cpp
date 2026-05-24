#include "HashApiEncoding.h"

namespace hashapi {
namespace {

constexpr char kBase64Chars[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/";

} // namespace

std::size_t base64EncodedLength(std::size_t in_len)
{
    const std::size_t full_groups = in_len / 3;
    const std::size_t remaining = in_len % 3;
    return full_groups * 4 + (remaining == 0 ? 0 : remaining + 1);
}

void base64EncodeInto(std::string& encoded, const std::uint8_t* bytes_to_encode, std::size_t in_len)
{
    encoded.clear();
    encoded.reserve(base64EncodedLength(in_len));

    std::size_t offset = 0;
    while (offset + 2 < in_len) {
        const std::uint32_t value =
            (static_cast<std::uint32_t>(bytes_to_encode[offset]) << 16) |
            (static_cast<std::uint32_t>(bytes_to_encode[offset + 1]) << 8) |
            static_cast<std::uint32_t>(bytes_to_encode[offset + 2]);
        encoded.push_back(kBase64Chars[(value >> 18) & 0x3f]);
        encoded.push_back(kBase64Chars[(value >> 12) & 0x3f]);
        encoded.push_back(kBase64Chars[(value >> 6) & 0x3f]);
        encoded.push_back(kBase64Chars[value & 0x3f]);
        offset += 3;
    }

    const std::size_t remaining = in_len - offset;
    if (remaining == 1) {
        const std::uint32_t value = static_cast<std::uint32_t>(bytes_to_encode[offset]) << 16;
        encoded.push_back(kBase64Chars[(value >> 18) & 0x3f]);
        encoded.push_back(kBase64Chars[(value >> 12) & 0x3f]);
    } else if (remaining == 2) {
        const std::uint32_t value =
            (static_cast<std::uint32_t>(bytes_to_encode[offset]) << 16) |
            (static_cast<std::uint32_t>(bytes_to_encode[offset + 1]) << 8);
        encoded.push_back(kBase64Chars[(value >> 18) & 0x3f]);
        encoded.push_back(kBase64Chars[(value >> 12) & 0x3f]);
        encoded.push_back(kBase64Chars[(value >> 6) & 0x3f]);
    }
}

std::string base64Encode(const std::uint8_t* bytes_to_encode, std::size_t in_len)
{
    std::string encoded;
    base64EncodeInto(encoded, bytes_to_encode, in_len);

    return encoded;
}

} // namespace hashapi
