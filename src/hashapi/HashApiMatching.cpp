#include "HashApiMatching.h"

#include <algorithm>
#include <cctype>
#include <regex>

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
    static const std::regex xuni_pattern(R"(XUNI\d)");
    return std::regex_search(hash, xuni_pattern);
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
