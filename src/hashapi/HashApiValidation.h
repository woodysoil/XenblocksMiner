#pragma once

#include "HashApiTypes.h"

#include <string>
#include <vector>

namespace hashapi {

bool isHexString(const std::string& value);
std::string normalizeHex(const std::string& value);
std::vector<std::string> validateRequest(const HashApiRequest& request);
bool isValidRequest(const HashApiRequest& request);
std::string joinErrors(const std::vector<std::string>& errors);

} // namespace hashapi
