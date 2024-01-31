#include "MerkleTree.h"
#include <openssl/evp.h>
#include <openssl/err.h>
#include <iostream>
#include <cstdio>

// Constructor implementation
MerkleTree::MerkleTree(const std::vector<std::string>& elements) {
    buildTree(elements);
}

// Returns the Merkle root hash
std::string MerkleTree::GetMerkleRoot() const {
    return merkleRoot;
}

// Prints the Merkle tree
void MerkleTree::PrintTree() const {
    for(const auto& item : merkleTree) {
        std::cout << item.first << " -> (" << item.second.first << ", " << item.second.second << ")\n";
    }
}

// OpenSSL error handling
void handleErrors(void) {
    ERR_print_errors_fp(stderr);
    abort();
}

// Hashes a string using SHA-256
std::string MerkleTree::hashValue(const std::string& value) const {
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

// Recursive function to build the Merkle tree
void MerkleTree::buildTree(const std::vector<std::string>& elements) {
    if(elements.size() == 1) {
        merkleRoot = elements[0];
        return;
    }

    std::vector<std::string> newElements;
    for(size_t i = 0; i < elements.size(); i += 2) {
        std::string left = elements[i];
        std::string right = (i + 1 < elements.size()) ? elements[i + 1] : left;
        std::string combined = left + right;
        std::string newHash = hashValue(combined);
        merkleTree[newHash] = std::make_pair(left, right);
        newElements.push_back(newHash);
    }

    buildTree(newElements);
}
