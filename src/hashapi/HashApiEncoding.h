#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace hashapi {

std::size_t base64EncodedLength(std::size_t in_len);
void base64EncodeInto(std::string& encoded, const std::uint8_t* bytes_to_encode, std::size_t in_len);
std::string base64Encode(const std::uint8_t* bytes_to_encode, std::size_t in_len);

} // namespace hashapi
