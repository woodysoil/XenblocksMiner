#pragma once

#include "HashApiTypes.h"

namespace hashapi {

class CpuHashBackend : public IHashBackend {
public:
    HashApiResult runBatch(const HashApiRequest& request) override;
};

} // namespace hashapi
