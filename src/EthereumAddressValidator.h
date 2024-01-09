// EthereumAddressValidator.h

#ifndef ETHEREUM_ADDRESS_VALIDATOR_H
#define ETHEREUM_ADDRESS_VALIDATOR_H

#include <string>

class EthereumAddressValidator {
public:
    bool isValid(const std::string& address);

private:
    std::string keccak_256(const std::string& input);
    std::string to_checksum_address(const std::string& address);
    bool is_valid_ethereum_address(const std::string& address);
};

#endif // ETHEREUM_ADDRESS_VALIDATOR_H
