#include "SHA256Hasher.h"
#include <openssl/evp.h>
#include <openssl/err.h>
#include <cstdio>

void SHA256Hasher::handleErrors() {
    ERR_print_errors_fp(stderr);
    abort();
}

std::string SHA256Hasher::sha256(const std::string& value) {
    unsigned char hash[EVP_MAX_MD_SIZE];
    unsigned int hash_len;
    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    if (!ctx) handleErrors();

    if (1 != EVP_DigestInit_ex(ctx, EVP_sha256(), NULL))
        handleErrors();

    if (1 != EVP_DigestUpdate(ctx, value.c_str(), value.size()))
        handleErrors();

    if (1 != EVP_DigestFinal_ex(ctx, hash, &hash_len))
        handleErrors();

    EVP_MD_CTX_free(ctx);

    std::string hashString;
    for (unsigned int i = 0; i < hash_len; i++) {
        char buf[3];
        sprintf(buf, "%02x", hash[i]);
        hashString += buf;
    }
    return hashString;
}
