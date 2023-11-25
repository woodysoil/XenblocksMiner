import sha3

def is_valid_ethereum_address(address):
    """
    Validate an Ethereum address using EIP-55 checksum.

    :param address: Ethereum address to validate.
    :return: True if valid, False otherwise.
    """
    # Check basic format
    if not isinstance(address, str) or not address.startswith("0x") or len(address) != 42:
        return False

    # Remove '0x' prefix and lowercase the address
    address_suffix = address[2:].lower()

    # Create a Keccak-256 (sha3) hash of the lowercase address
    keccak_hash = sha3.keccak_256(address_suffix.encode()).hexdigest()

    # Check each character
    for i in range(40):
        if address[i + 2].lower() == address_suffix[i]:
            # If the character is a digit, it's fine
            if address_suffix[i] in '0123456789':
                continue
            # If the character is a letter, check EIP-55 checksum
            if keccak_hash[i] < '8' and address[i + 2].isupper():
                return False
            if keccak_hash[i] >= '8' and address[i + 2].islower():
                return False
        else:
            return False

    return True

# Test the function
address = "0x24691E54aFafe2416a8252097C9Ca67557271475"  # Replace with an Ethereum address
if is_valid_ethereum_address(address):
    print("Valid Ethereum address")
else:
    print("Invalid Ethereum address")
