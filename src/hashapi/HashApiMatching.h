#pragma once

#include "HashApiTypes.h"

#include <cstddef>
#include <string>

namespace hashapi {

bool isSuperblockHash(const std::string& hash);
bool hasXuniMatch(const std::string& hash);

void appendMatches(const HashApiRequest& request,
                   HashApiResult& result,
                   const std::string& key,
                   const std::string& hash,
                   std::size_t attempt_index);

} // namespace hashapi
