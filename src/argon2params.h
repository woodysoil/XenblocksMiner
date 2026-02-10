#pragma once

#include <cstdint>
#include <string>
#include "argon2-common.h"

class Argon2Params
{
private:
    argon2::Type type;
    argon2::Version version;
    void* secret, * ad;
    std::string salt;
    std::uint32_t outLen, secretLen, adLen;
    std::uint32_t t_cost, m_cost, lanes;

    std::uint32_t segmentBlocks;

    static void digestLong(void* out, std::size_t outLen,
        const void* in, std::size_t inLen);

    void initialHash(void* out, const void* pwd, std::size_t pwdLen) const;

public:
    std::uint32_t getOutputLength() const { return outLen; }

    const std::string getSalt() const { return salt; }
    std::uint32_t getSaltLength() const { return salt.length(); }

    const void* getSecret() const { return secret; }
    std::uint32_t getSecretLength() const { return secretLen; }

    const void* getAssocData() const { return ad; }
    std::uint32_t getAssocDataLength() const { return adLen; }

    std::uint32_t getTimeCost() const { return t_cost; }
    std::uint32_t getMemoryCost() const { return m_cost; }
    std::uint32_t getLanes() const { return lanes; }

    std::uint32_t getSegmentBlocks() const { return segmentBlocks; }
    std::uint32_t getLaneBlocks() const {
        return segmentBlocks * argon2::ARGON2_SYNC_POINTS;
    }
    std::uint32_t getMemoryBlocks() const { return getLaneBlocks() * lanes; }
    std::size_t getMemorySize() const {
        return static_cast<std::size_t>(getMemoryBlocks()) * argon2::ARGON2_BLOCK_SIZE;
    }
    argon2::Type getType() const { return type; }

    argon2::Version getVersion() const { return version; }

    Argon2Params() {}

    Argon2Params(
        argon2::Type type, argon2::Version version,
        std::size_t outLen, std::string salt,
        void* secret, std::size_t secretLen,
        void* ad, std::size_t adLen,
        std::size_t t_cost, std::size_t m_cost, std::size_t lanes);

    void fillFirstBlocks(void* memory, const void* pwd, std::size_t pwdLen) const;

    void finalize(void* out, const void* memory) const;
};
