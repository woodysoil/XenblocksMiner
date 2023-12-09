#ifndef SHA256HASHER_H
#define SHA256HASHER_H

#include <string>

class SHA256Hasher {
public:
    // Generates a SHA-256 hash of the given string.
    static std::string sha256(const std::string& value);

private:
    // Handles OpenSSL errors.
    static void handleErrors();
};

#endif // SHA256HASHER_H
