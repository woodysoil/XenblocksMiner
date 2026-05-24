#pragma once

#include "HashApiTypes.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

class ComputeBackend;
class Argon2Params;

namespace hashapi {

class CudaHashBackend : public IHashBackend {
public:
    explicit CudaHashBackend(ComputeBackend& backend);
    explicit CudaHashBackend(std::unique_ptr<ComputeBackend> backend);
    ~CudaHashBackend() override;

    HashApiResult runBatch(const HashApiRequest& request) override;

private:
    ComputeBackend& backend();
    const ComputeBackend& backend() const;
    void ensureInitialized(ComputeBackend& backend,
                           const Argon2Params& params,
                           std::size_t batch_size);

    ComputeBackend* backend_ = nullptr;
    std::unique_ptr<ComputeBackend> owned_backend_;
    std::vector<std::string> password_storage_;
    bool initialized_ = false;
    std::size_t initialized_batch_size_ = 0;
    std::uint32_t initialized_segment_blocks_ = 0;
};

} // namespace hashapi
