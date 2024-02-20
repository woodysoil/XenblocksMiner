#include "EthereumSignatureValidator.h"

#include <cryptopp/hex.h>
#include <cryptopp/keccak.h>
#include <openssl/ec.h>
#include <openssl/ecdsa.h>
#include <openssl/obj_mac.h>
#include <openssl/sha.h>
#include <secp256k1.h>
#include <secp256k1_recovery.h>

bool EthereumSignatureValidator::verifyMessage(
    const std::string &message, const std::string &signatureHex,
    const std::string &expectedAddress) {
    using namespace CryptoPP;
    secp256k1_context *context =
        secp256k1_context_create(SECP256K1_CONTEXT_VERIFY);

    // 1. Prepare the message
    std::string prefix = "\x19"
                         "Ethereum Signed Message:\n" +
                         std::to_string(message.length()) + message;
    CryptoPP::Keccak_256 keccak1;
    byte digest[CryptoPP::Keccak_256::DIGESTSIZE];
    keccak1.CalculateDigest(digest, (const byte *)prefix.data(), prefix.size());

    // 2. Decode the signature
    std::string signatureBin;
    HexDecoder decoder;
    std::string signatureHexWithoutPrefix = signatureHex;
    if (signatureHex.size() >= 2 && signatureHex[0] == '0' &&
        signatureHex[1] == 'x') {
        signatureHexWithoutPrefix = signatureHex.substr(2); // remove "0x"
    }

    decoder.Put((byte *)signatureHexWithoutPrefix.data(),
                signatureHexWithoutPrefix.size());
    decoder.MessageEnd();
    signatureBin.resize(decoder.MaxRetrievable());
    decoder.Get((byte *)signatureBin.data(), signatureBin.size());

    // Split signature into r, s and v
    if (signatureBin.size() < 64)
        return false; // Invalid signature size
    const byte *sig = (const byte *)signatureBin.data();
    int v = sig[64];
    if (v == 27 || v == 28) {
        v -= 27;
    }

    // 3. Recover the public key
    secp256k1_ecdsa_recoverable_signature recSig;
    if (!secp256k1_ecdsa_recoverable_signature_parse_compact(context, &recSig,
                                                             &sig[0], v)) {
        secp256k1_context_destroy(context);
        return false; // Failed to parse signature
    }

    secp256k1_pubkey pubkey;
    if (!secp256k1_ecdsa_recover(context, &pubkey, &recSig, digest)) {
        secp256k1_context_destroy(context);
        return false; // Failed to recover public key
    }

    // 4. Serialize the recovered public key
    unsigned char serializedPubkey[65];
    size_t serializedPubkeyLength = sizeof(serializedPubkey);
    secp256k1_ec_pubkey_serialize(context, serializedPubkey,
                                  &serializedPubkeyLength, &pubkey,
                                  SECP256K1_EC_UNCOMPRESSED);

    // 5. Calculate the Ethereum address using Keccak-256
    Keccak_256 keccak;
    byte keccakDigest[Keccak_256::DIGESTSIZE];
    keccak.CalculateDigest(keccakDigest, serializedPubkey + 1, 64);

    std::string address = "0x";
    HexEncoder encoder(new StringSink(address));
    encoder.Put(keccakDigest + (Keccak_256::DIGESTSIZE - 20),
                20); // Last 20 bytes
    encoder.MessageEnd();

    // Convert the address to lower case for comparison
    std::transform(address.begin(), address.end(), address.begin(), ::tolower);

    // 6. Compare with the expected address
    // Ensure the expected address is also lower case
    std::string expectedAddressLowerCase = expectedAddress;
    std::transform(expectedAddressLowerCase.begin(),
                   expectedAddressLowerCase.end(),
                   expectedAddressLowerCase.begin(), ::tolower);
    secp256k1_context_destroy(context);
    return expectedAddressLowerCase == address;
}
