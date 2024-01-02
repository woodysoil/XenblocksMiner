#include "argon2params.h"

#include "blake2b.h"

#include <cstring>
#include <algorithm>
#include <string>



static void store32(void *dst, std::uint32_t v)
{
    auto out = static_cast<std::uint8_t *>(dst);
    *out++ = static_cast<std::uint8_t>(v); v >>= 8;
    *out++ = static_cast<std::uint8_t>(v); v >>= 8;
    *out++ = static_cast<std::uint8_t>(v); v >>= 8;
    *out++ = static_cast<std::uint8_t>(v);
}

Argon2Params::Argon2Params(
    argon2::Type type,
    argon2::Version version,
        std::size_t outLen,
        std::string salt, 
        void *secret, std::size_t secretLen,
        void *ad, std::size_t adLen,
        std::size_t t_cost, std::size_t m_cost, std::size_t lanes)
    : type(type), version(version),
    salt(salt), secret(secret), ad(ad),
    outLen(outLen), secretLen(secretLen), adLen(adLen),
    t_cost(t_cost), m_cost(m_cost), lanes(lanes)
{
    // TODO validate inputs
    std::size_t segments = lanes * argon2::ARGON2_SYNC_POINTS;
    segmentBlocks = std::max(m_cost, 2 * segments) / segments;
}

void Argon2Params::digestLong(void *out, std::size_t outLen,
                              const void *in, std::size_t inLen)
{
    auto bout = static_cast<std::uint8_t *>(out);
    std::uint8_t outlen_bytes[sizeof(std::uint32_t)];
    Blake2b blake;

    store32(outlen_bytes, static_cast<std::uint32_t>(outLen));
    if (outLen <= Blake2b::OUT_BYTES) {
        blake.init(outLen);
        blake.update(outlen_bytes, sizeof(outlen_bytes));
        blake.update(in, inLen);
        blake.final(out, outLen);
    } else {
        std::uint8_t out_buffer[Blake2b::OUT_BYTES];

        blake.init(Blake2b::OUT_BYTES);
        blake.update(outlen_bytes, sizeof(outlen_bytes));
        blake.update(in, inLen);
        blake.final(out_buffer, Blake2b::OUT_BYTES);

        std::memcpy(bout, out_buffer, Blake2b::OUT_BYTES / 2);
        bout += Blake2b::OUT_BYTES / 2;

        std::size_t toProduce = outLen - Blake2b::OUT_BYTES / 2;
        while (toProduce > Blake2b::OUT_BYTES) {
            blake.init(Blake2b::OUT_BYTES);
            blake.update(out_buffer, Blake2b::OUT_BYTES);
            blake.final(out_buffer, Blake2b::OUT_BYTES);

            std::memcpy(bout, out_buffer, Blake2b::OUT_BYTES / 2);
            bout += Blake2b::OUT_BYTES / 2;
            toProduce -= Blake2b::OUT_BYTES / 2;
        }

        blake.init(toProduce);
        blake.update(out_buffer, Blake2b::OUT_BYTES);
        blake.final(bout, toProduce);
    }
}

std::string hex_to_bytes(const std::string& hex) {
    std::string bytes;
    for (std::size_t i = 0; i < hex.length(); i += 2) {
        std::string byteString = hex.substr(i, 2);
        char byte = (char)std::stoi(byteString, nullptr, 16);
        bytes.push_back(byte);
    }
    return bytes;
}

void Argon2Params::initialHash(
        void *out, const void *pwd, std::size_t pwdLen) const
{
    Blake2b blake;
    std::uint8_t value[sizeof(std::uint32_t)];

    blake.init(argon2::ARGON2_PREHASH_DIGEST_LENGTH);

    store32(value, lanes);      blake.update(value, sizeof(value));
    store32(value, outLen);     blake.update(value, sizeof(value));
    store32(value, m_cost);     blake.update(value, sizeof(value));
    store32(value, t_cost);     blake.update(value, sizeof(value));
    store32(value, version);    blake.update(value, sizeof(value));
    store32(value, type);       blake.update(value, sizeof(value));
    store32(value, pwdLen);     blake.update(value, sizeof(value));
    blake.update(pwd, pwdLen);

    std::string hexSalt = hex_to_bytes(salt);
    store32(value, hexSalt.length());    blake.update(value, sizeof(value));
    blake.update(hexSalt.c_str(), hexSalt.length());

    store32(value, secretLen);  blake.update(value, sizeof(value));
    blake.update(secret, secretLen);
    store32(value, adLen);      blake.update(value, sizeof(value));
    blake.update(ad, adLen);

    blake.final(out, argon2::ARGON2_PREHASH_DIGEST_LENGTH);
}

void Argon2Params::fillFirstBlocks(
        void *memory, const void *pwd, std::size_t pwdLen) const
{
    std::uint8_t initHash[argon2::ARGON2_PREHASH_SEED_LENGTH];
    initialHash(initHash, pwd, pwdLen);

    auto bmemory = static_cast<std::uint8_t *>(memory);

    store32(initHash + argon2::ARGON2_PREHASH_DIGEST_LENGTH, 0);
    for (std::uint32_t l = 0; l < lanes; l++) {
        store32(initHash + argon2::ARGON2_PREHASH_DIGEST_LENGTH + 4, l);
        digestLong(bmemory, argon2::ARGON2_BLOCK_SIZE, initHash, sizeof(initHash));
        bmemory += argon2::ARGON2_BLOCK_SIZE;
    }

    store32(initHash + argon2::ARGON2_PREHASH_DIGEST_LENGTH, 1);
    for (std::uint32_t l = 0; l < lanes; l++) {
        store32(initHash + argon2::ARGON2_PREHASH_DIGEST_LENGTH + 4, l);
        digestLong(bmemory, argon2::ARGON2_BLOCK_SIZE, initHash, sizeof(initHash));
        bmemory += argon2::ARGON2_BLOCK_SIZE;
    }
}

void Argon2Params::finalize(void *out, const void *memory) const
{
    struct block {
        std::uint64_t v[argon2::ARGON2_BLOCK_SIZE / 8];
    };

    auto cursor = static_cast<const block *>(memory);

    block xored = *cursor;
    for (std::uint32_t l = 1; l < lanes; l++) {
        ++cursor;
        for (std::size_t i = 0; i < argon2::ARGON2_BLOCK_SIZE / 8; i++) {
            xored.v[i] ^= cursor->v[i];
        }
    }

    digestLong(out, outLen, &xored, argon2::ARGON2_BLOCK_SIZE);
}
