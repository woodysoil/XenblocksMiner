#include <iostream>
#include <string>
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <argon2.h>
#include <vector>
#include <unordered_map>
#include <openssl/evp.h>
#include <openssl/err.h>

// Callback function writes data to a std::string, and will be called by libcurl as data arrives
static size_t WriteCallback(void *contents, size_t size, size_t nmemb, std::string *s) {
    size_t newLength = size * nmemb;
    try {
        s->append((char*)contents, newLength);
    } catch(std::bad_alloc &e) {
        // Handle memory problem
        return 0;
    }
    return newLength;
}

// HTTP GET request function
std::string HttpGetRequest(const std::string& url, long timeout) {
    CURL *curl;
    CURLcode res;
    std::string readBuffer;
    curl = curl_easy_init();
    if(curl) {
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, timeout);
        res = curl_easy_perform(curl);
        if (res != CURLE_OK) {
            std::cerr << "curl_easy_perform() failed: " << curl_easy_strerror(res) << std::endl;
            // Handle error or throw an exception
        }
        curl_easy_cleanup(curl);
    }
    return readBuffer;
}

std::string HttpPostRequest(const std::string& url, const nlohmann::json& payload, long timeout) {
    CURL *curl;
    CURLcode res;
    std::string readBuffer;
    struct curl_slist *headers = nullptr;

    curl = curl_easy_init();
    if(curl) {
        headers = curl_slist_append(headers, "Content-Type: application/json");
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        std::string json_payload = payload.dump();
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_payload.c_str());
        curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, json_payload.size());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, timeout);
        // curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L); // detail output

        res = curl_easy_perform(curl);
        if (res != CURLE_OK) {
            std::cerr << "curl_easy_perform() failed: " << curl_easy_strerror(res) << std::endl;
        }
        curl_slist_free_all(headers);  // free the header list
        curl_easy_cleanup(curl);
    }
    return readBuffer;
}

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

// Define the salt and hash length
const size_t SALT_LEN = 64; // 64 bytes for salt
const size_t HASH_LEN = 64; // 64 bytes for hash

// Helper function to convert a single hex character to a byte
uint8_t hexchar_to_byte(char ch) {
    if (ch >= '0' && ch <= '9') {
        return ch - '0';
    } else if (ch >= 'a' && ch <= 'f') {
        return ch - 'a' + 10;
    } else if (ch >= 'A' && ch <= 'F') {
        return ch - 'A' + 10;
    } else {
        throw std::invalid_argument("Invalid hexadecimal character");
    }
}

// Function to convert a hex string to a byte vector
std::vector<uint8_t> hexstr_to_bytes(const std::string& hexstr) {
    std::vector<uint8_t> bytes;

    for (size_t i = 0; i < hexstr.size(); i += 2) {
        bytes.push_back((hexchar_to_byte(hexstr[i]) << 4) | hexchar_to_byte(hexstr[i + 1]));
    }

    return bytes;
}

// Function to generate an Argon2id hash
std::string generate_argon2id_hash(const std::string& password) {
    // Define Argon2id parameters
    const uint32_t t_cost = 1;  // Time cost
    const uint32_t m_cost = 1 << 16; // Memory cost
    const uint32_t parallelism = 1;  // Parallelism degree

    // Generate a salt (for this example, we'll just use a fixed salt)
    std::string hex_string = "24691E54aFafe2416a8252097C9Ca67557271475";
    std::vector<uint8_t> salt = hexstr_to_bytes(hex_string);

    // Prepare output buffer
    size_t encoded_len = argon2_encodedlen(t_cost, m_cost, parallelism, SALT_LEN, HASH_LEN, Argon2_id);
    std::vector<char> encoded_hash(encoded_len);

    // Generate the hash
    int result = argon2id_hash_encoded(t_cost, m_cost, parallelism, password.data(), password.size(), salt.data(), salt.size(), HASH_LEN, encoded_hash.data(), encoded_hash.size());
    if (result != ARGON2_OK) {
        throw std::runtime_error("Failed to generate Argon2id hash");
    }

    return std::string(encoded_hash.begin(), encoded_hash.end());
}

// Function to verify a password using Argon2id
bool verify_argon2id_hash(const std::string& password, const std::string& hashed_password) {
    if (argon2id_verify(hashed_password.c_str(), password.c_str(), password.length()) == ARGON2_OK) {
        return true; // Password matches
    }
    return false; // Password does not match
}

void submit_pow(const std::string& account_address, const std::string& key, const std::string& hash_to_verify) {
    const std::string url = "http://xenminer.mooo.com:4445/getblocks/lastblock";
    std::string response = HttpGetRequest(url, 10); // 10 seconds timeout

    if (response.empty()) {
        std::cerr << "Failed to get data from the server." << std::endl;
        return;
    }

    try {
        auto records = nlohmann::json::parse(response);
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

            if (verify_argon2id_hash(record_key, record_hash_to_verify)) {
                verified_hashes.push_back(hashValue(std::to_string(block_id) + record_hash_to_verify + record_key + account));
            }
        }

        if (!verified_hashes.empty()) {
            auto result = buildMerkleTree(verified_hashes);
            std::string merkleRoot = result.first;

            int output_block_id = block_id / 100;

            nlohmann::json payload = {
                {"account_address", account_address},
                {"block_id", output_block_id},
                {"merkle_root", merkleRoot},
                {"key", key},
                {"hash_to_verify", hash_to_verify}
            };

            std::cout << "Payload: " << payload.dump(4) << std::endl;

            std::string response = HttpPostRequest("http://xenblocks.io:4446/send_pow", payload, 10); // 10 seconds timeout

            std::cout << "Server Response: " << response << std::endl;

            try {
                auto jsonResponse = nlohmann::json::parse(response);
                std::cout << "Server Response: " << jsonResponse.dump(4) << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "Failed to parse JSON response: " << e.what() << std::endl;
            }

        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to parse JSON response or process data: " << e.what() << std::endl;
        return;
    }
}

int main() {

    submit_pow("...", "...", "...");

    return 0;
}
