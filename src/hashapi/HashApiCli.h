#pragma once

namespace hashapi {

bool isHashApiCommand(int argc, const char* const* argv);
int runHashApiCli(int argc, const char* const* argv);

} // namespace hashapi
