#ifndef MERKLETREE_H
#define MERKLETREE_H

#include <string>
#include <vector>
#include <unordered_map>

class MerkleTree {
public:
    MerkleTree(const std::vector<std::string>& elements);
    std::string GetMerkleRoot() const;
    void PrintTree() const;

private:
    std::string hashValue(const std::string& value) const;
    void buildTree(const std::vector<std::string>& elements);

    std::unordered_map<std::string, std::pair<std::string, std::string>> merkleTree;
    std::string merkleRoot;
};

#endif // MERKLETREE_H
