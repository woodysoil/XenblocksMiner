#include "HashApiCli.h"

int main(int argc, const char* const* argv)
{
    if (argc <= 1) {
        const char* help_argv[] = {"hashapi-cli", "hash-help"};
        return hashapi::runHashApiCli(2, help_argv);
    }
    return hashapi::runHashApiCli(argc, argv);
}
