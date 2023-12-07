// EthereumAddressValidator.cpp

#include "EthereumAddressValidator.h"
#include <cryptopp/keccak.h>
#include <cryptopp/hex.h>
#include <regex>
#include <algorithm>

// check if an Ethereum address is valid
bool EthereumAddressValidator::isValid(const std::string& address) {
    return is_valid_ethereum_address(address);
}

// Function to calculate Keccak-256 hash
std::string EthereumAddressValidator::keccak_256(const std::string& input) {
    CryptoPP::Keccak_256 hash;
    std::string digest;

    CryptoPP::StringSource ss(input, true,
        new CryptoPP::HashFilter(hash,
            new CryptoPP::HexEncoder(
                new CryptoPP::StringSink(digest)
            )
        )
    );

    return digest;
}

// Function to convert an address to EIP-55 checksum address
std::string EthereumAddressValidator::to_checksum_address(const std::string& address) {
    std::string address_lower = address.substr(2);
    std::transform(address_lower.begin(), address_lower.end(), address_lower.begin(), ::tolower);
    std::string hash = keccak_256(address_lower);

    std::string checksum_address = "0x";
    for (size_t i = 0; i < address_lower.size(); ++i) {
        char c = address_lower[i];
        if (hash[i] >= '8') {
            checksum_address += toupper(c);
        } else {
            checksum_address += c;
        }
    }
    return checksum_address;
}

// Function to check if an Ethereum address is valid
bool EthereumAddressValidator::is_valid_ethereum_address(const std::string& address) {
    // Basic hexadecimal pattern check
    if (!std::regex_match(address, std::regex("^0x[0-9a-fA-F]{40}$"))) {
        return false;
    }

    // EIP-55 checksum encoding check
    try {
        return address == to_checksum_address(address);
    } catch (...) {
        return false;
    }
}