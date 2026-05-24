#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace hashapi {

std::string base64Encode(const std::uint8_t* bytes_to_encode, std::size_t in_len);

} // namespace hashapi
