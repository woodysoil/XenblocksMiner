#pragma once

#include "HashApiTypes.h"

#include <string>

namespace hashapi {

std::string toJson(const HashApiMatch& match);
std::string toJson(const HashApiTimings& timings);
std::string toJson(const HashApiResult& result);

} // namespace hashapi
