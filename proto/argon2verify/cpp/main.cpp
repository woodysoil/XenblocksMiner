#include <argon2.h>
#include <iostream>
#include <string>
#include <vector>

// Define the salt and hash length
const size_t SALT_LEN = 64; // 64 bytes for salt
const size_t HASH_LEN = 64; // 64 bytes for hash

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

// Function to convert a hex string to a byte vector
std::vector<uint8_t> hexstr_to_bytes(const std::string& hexstr) {
    std::vector<uint8_t> bytes;

    for (size_t i = 0; i < hexstr.size(); i += 2) {
        bytes.push_back((hexchar_to_byte(hexstr[i]) << 4) | hexchar_to_byte(hexstr[i + 1]));
    }

    return bytes;
}

// Function to generate an Argon2id hash
std::string generate_argon2id_hash(const std::string& password) {
    // Define Argon2id parameters
    const uint32_t t_cost = 1;  // Time cost
    const uint32_t m_cost = 1 << 16; // Memory cost
    const uint32_t parallelism = 1;  // Parallelism degree

    // Generate a salt (for this example, we'll just use a fixed salt)
    std::string hex_string = "24691E54aFafe2416a8252097C9Ca67557271475";
    std::vector<uint8_t> salt = hexstr_to_bytes(hex_string);

    // Prepare output buffer
    size_t encoded_len = argon2_encodedlen(t_cost, m_cost, parallelism, SALT_LEN, HASH_LEN, Argon2_id);
    std::vector<char> encoded_hash(encoded_len);

    // Generate the hash
    int result = argon2id_hash_encoded(t_cost, m_cost, parallelism, password.data(), password.size(), salt.data(), salt.size(), HASH_LEN, encoded_hash.data(), encoded_hash.size());
    if (result != ARGON2_OK) {
        throw std::runtime_error("Failed to generate Argon2id hash");
    }

    return std::string(encoded_hash.begin(), encoded_hash.end());
}

// Function to verify a password using Argon2id
bool verify_argon2id_hash(const std::string& password, const std::string& hashed_password) {
    if (argon2id_verify(hashed_password.c_str(), password.c_str(), password.length()) == ARGON2_OK) {
        return true; // Password matches
    }
    return false; // Password does not match
}

int main() {
    std::string password = "woody"; // The password to be hashed and verified

    // Generate an Argon2id hash
    std::string hashed_password = generate_argon2id_hash(password);

    std::cout << "Generated Hash: " << hashed_password << std::endl;

    // Verify the password
    if (verify_argon2id_hash(password, hashed_password)) {
        std::cout << "Password verified!" << std::endl;
    } else {
        std::cout << "Invalid password!" << std::endl;
    }

    return 0;
}
