#include "PowSubmitter.h"
#include "HttpClient.h"
#include "MerkleTree.h"
#include "Argon2idHasher.h"
#include "SHA256Hasher.h"
#include <nlohmann/json.hpp>
#include <iostream>
#include <string>

void PowSubmitter::submitPow(const std::string& account_address, const std::string& key, const std::string& hash_to_verify) {
    HttpClient httpClient;

    HttpResponse lastBlockResponse = httpClient.HttpGet("http://xenminer.mooo.com:4445/getblocks/lastblock", 10);
    if (lastBlockResponse.GetStatusCode() != 200) {
        std::cerr << "Failed to get data from the server." << std::endl;
        return;
    }

    try {
        auto records = nlohmann::json::parse(lastBlockResponse.GetBody());
        std::vector<std::string> verified_hashes;
        std::string account, record_hash_to_verify, record_key;
        int block_id;

        for (const auto& record : records) {
            block_id = record.value("block_id", 0);
            record_hash_to_verify = record.value("hash_to_verify", "");
            record_key = record.value("key", "");
            account = record.value("account", "");

            if (record_key.empty() || record_hash_to_verify.empty()) {
                std::cout << "Skipping record due to None value(s)." << std::endl;
                continue;
            }

            if (Argon2idHasher::verifyHash(record_key, record_hash_to_verify)) {
                verified_hashes.push_back(SHA256Hasher::sha256(std::to_string(block_id) + record_hash_to_verify + record_key + account));
            }
        }

        if (!verified_hashes.empty()) {
            MerkleTree resultTree(verified_hashes);
            std::string merkleRoot = resultTree.GetMerkleRoot();

            int output_block_id = block_id / 100;

            nlohmann::json payload = {
                {"account_address", account_address},
                {"block_id", output_block_id},
                {"merkle_root", merkleRoot},
                {"key", key},
                {"hash_to_verify", hash_to_verify}
            };

            std::cout << "Payload: " << payload.dump(4) << std::endl;

            HttpResponse powResponse = httpClient.HttpPost("http://xenblocks.io:4446/send_pow", payload, 10);
            std::cout << "Server Response: " << powResponse.GetBody() << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to parse JSON response or process data: " << e.what() << std::endl;
    }
}
