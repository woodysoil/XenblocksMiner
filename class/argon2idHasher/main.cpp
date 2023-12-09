// main.cpp

#include "Argon2idHasher.h"
#include <iostream>

int main() {
    uint32_t t_cost = 1;       // Time cost
    uint32_t m_cost = 1 << 16; // Memory cost
    uint32_t parallelism = 1;  // Parallelism degree
    std::string salt_hex = "24691E54aFafe2416a8252097C9Ca67557271475"; // Hex string for salt
    size_t hash_len = 64;      // Hash length

    Argon2idHasher hasher(t_cost, m_cost, parallelism, salt_hex, hash_len);
    std::string password = "woody"; // The password to be hashed and verified

    // Generate hash
    std::string hashed_password = hasher.generateHash(password);
    std::cout << "Generated Hash: " << hashed_password << std::endl;

    // Verify hash
    if (Argon2idHasher::verifyHash(password, hashed_password)) {
        std::cout << "Password verified!" << std::endl;
    } else {
        std::cout << "Invalid password!" << std::endl;
    }

    return 0;
}
