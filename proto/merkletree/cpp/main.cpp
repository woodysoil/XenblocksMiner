#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <openssl/evp.h>
#include <openssl/err.h>

void handleErrors(void) {
    ERR_print_errors_fp(stderr);
    abort();
}

std::string hashValue(const std::string& value) {
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

std::pair<std::string, std::unordered_map<std::string, std::pair<std::string, std::string>>> 
buildMerkleTree(const std::vector<std::string>& elements, 
                std::unordered_map<std::string, std::pair<std::string, std::string>> merkleTree = {}) 
{
    if(elements.size() == 1) {
        return std::make_pair(elements[0], merkleTree);
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

    return buildMerkleTree(newElements, merkleTree);
}

int main() {
    std::vector<std::string> verifiedHashes = {"hash1", "hash2", "hash3"};
    auto result = buildMerkleTree(verifiedHashes);
    std::string merkleRoot = result.first;
    std::unordered_map<std::string, std::pair<std::string, std::string>> merkleTree = result.second;

    std::cout << "Merkle Root: " << merkleRoot << std::endl;

    for(const auto& item : merkleTree) {
        std::cout << item.first << " -> (" << item.second.first << ", " << item.second.second << ")\n";
    }

    return 0;
}