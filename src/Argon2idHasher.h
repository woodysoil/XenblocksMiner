// Argon2idHasher.h

#ifndef ARGON2ID_HASHER_H
#define ARGON2ID_HASHER_H

#include <cstdint>
#include <string>
#include <vector>

class Argon2idHasher {
public:
    Argon2idHasher(uint32_t t_cost, uint32_t m_cost, uint32_t parallelism, const std::string& salt_hex, size_t hash_len);

    std::string generateHash(const std::string& password);
    static bool verifyHash(const std::string& password, const std::string& hash);

private:
    uint32_t t_cost_;
    uint32_t m_cost_;
    uint32_t parallelism_;
    std::vector<uint8_t> salt_;
    size_t hash_len_;

    std::vector<uint8_t> hexstr_to_bytes(const std::string& hexstr);
};

#endif // ARGON2ID_HASHER_H
