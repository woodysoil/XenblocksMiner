// main.cpp

#include "EthereumAddressValidator.h"
#include <iostream>

int main() {
    EthereumAddressValidator validator;
    std::string address = "0x24691E54aFafe2416a8252097C9Ca67557271475";

    if (validator.isValid(address)) {
        std::cout << "Valid Ethereum address" << std::endl;
    } else {
        std::cout << "Invalid Ethereum address" << std::endl;
    }

    return 0;
}
