#include "SHA256Hasher.h"
#include <iostream>

int main() {
    std::string myValue = "Hello, world!";
    std::string myHash = SHA256Hasher::sha256(myValue);
    std::cout << "SHA-256: " << myHash << std::endl;
    return 0;
}
