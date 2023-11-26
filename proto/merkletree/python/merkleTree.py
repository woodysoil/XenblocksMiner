import hashlib

def hash_value(value):
    return hashlib.sha256(value.encode()).hexdigest()

def build_merkle_tree(elements, merkle_tree={}):
    if len(elements) == 1:
        return elements[0], merkle_tree

    new_elements = []
    for i in range(0, len(elements), 2):
        left = elements[i]
        right = elements[i + 1] if i + 1 < len(elements) else left
        combined = left + right
        new_hash = hash_value(combined)
        merkle_tree[new_hash] = (left, right)
        new_elements.append(new_hash)

    return build_merkle_tree(new_elements, merkle_tree)

def main():
    verified_hashes = ["hash1", "hash2", "hash3"]
    merkle_root, merkle_tree = build_merkle_tree(verified_hashes)

    print("Merkle Root:", merkle_root)
    for key, value in merkle_tree.items():
        print(f"{key} -> ({value[0]}, {value[1]})")

if __name__ == "__main__":
    main()
