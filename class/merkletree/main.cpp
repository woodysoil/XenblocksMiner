#include <iostream>
#include "MerkleTree.h"

int main() {
    std::vector<std::string> verifiedHashes = {"hash1", "hash2", "hash3"};
    MerkleTree tree(verifiedHashes);

    std::cout << "Merkle Root: " << tree.GetMerkleRoot() << std::endl;
    tree.PrintTree();

    return 0;
}
