#include "HashApiMatching.h"

#include <algorithm>
#include <cctype>

namespace hashapi {

bool isSuperblockHash(const std::string& hash)
{
    const auto uppercase_count = std::count_if(hash.begin(), hash.end(), [](unsigned char ch) {
        return std::isupper(ch) != 0;
    });
    return uppercase_count >= 50;
}

bool hasXuniMatch(const std::string& hash)
{
    constexpr const char* kXuniPrefix = "XUNI";
    constexpr std::size_t kXuniPrefixLength = 4;

    for (std::size_t offset = hash.find(kXuniPrefix);
         offset != std::string::npos;
         offset = hash.find(kXuniPrefix, offset + 1)) {
        const std::size_t digit_offset = offset + kXuniPrefixLength;
        if (digit_offset < hash.size() &&
            std::isdigit(static_cast<unsigned char>(hash[digit_offset])) != 0) {
            return true;
        }
    }
    return false;
}

void appendMatches(const HashApiRequest& request,
                   HashApiResult& result,
                   const std::string& key,
                   const std::string& hash,
                   std::size_t attempt_index)
{
    if (hash.find(request.target_pattern) != std::string::npos) {
        result.matches.push_back({
            key,
            hash,
            request.target_pattern,
            attempt_index,
            isSuperblockHash(hash),
        });
    }

    if (request.allow_xuni && hasXuniMatch(hash)) {
        result.matches.push_back({
            key,
            hash,
            "XUNI",
            attempt_index,
            false,
        });
    }
}

} // namespace hashapi
