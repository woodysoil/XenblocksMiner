// Argon2idHasher.cpp

#include "Argon2idHasher.h"
#include <argon2.h>
#include <stdexcept>
#include <algorithm>

Argon2idHasher::Argon2idHasher(uint32_t t_cost, uint32_t m_cost, uint32_t parallelism, const std::string& salt_hex, size_t hash_len)
    : t_cost_(t_cost), m_cost_(m_cost), parallelism_(parallelism), hash_len_(hash_len) {
    salt_ = hexstr_to_bytes(salt_hex);
}

std::string Argon2idHasher::generateHash(const std::string& password) {
    size_t encoded_len = argon2_encodedlen(t_cost_, m_cost_, parallelism_, salt_.size(), hash_len_, Argon2_id);
    std::vector<char> encoded_hash(encoded_len);

    int result = argon2id_hash_encoded(t_cost_, m_cost_, parallelism_, password.data(), password.size(), salt_.data(), salt_.size(), hash_len_, encoded_hash.data(), encoded_hash.size());
    if (result != ARGON2_OK) {
        throw std::runtime_error("Failed to generate Argon2id hash");
    }

    return std::string(encoded_hash.begin(), encoded_hash.end());
}

bool Argon2idHasher::verifyHash(const std::string& password, const std::string& hash) {
    return argon2id_verify(hash.c_str(), password.c_str(), password.length()) == ARGON2_OK;
}

// Helper function to convert a single hex character to a byte
uint8_t hexchar_to_byte(char ch) {
    if (ch >= '0' && ch <= '9') {
        return ch - '0';
    } else if (ch >= 'a' && ch <= 'f') {
        return ch - 'a' + 10;
    } else if (ch >= 'A' && ch <= 'F') {
        return ch - 'A' + 10;
    } else {
        throw std::invalid_argument("Invalid hexadecimal character");
    }
}

std::vector<uint8_t> Argon2idHasher::hexstr_to_bytes(const std::string& hexstr) {
    std::vector<uint8_t> bytes;
    for (size_t i = 0; i < hexstr.size(); i += 2) {
        bytes.push_back((hexchar_to_byte(hexstr[i]) << 4) | hexchar_to_byte(hexstr[i + 1]));
    }
    return bytes;
}