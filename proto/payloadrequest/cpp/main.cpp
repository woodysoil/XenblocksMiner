#include <iostream>
#include <string>
#include <curl/curl.h>
#include <nlohmann/json.hpp>

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

int main() {
    std::string hashed_data = "$argon2id$v=19$m=80,t=1,p=1$WEVOMTAwODIwMjJYRU4$StL5GRjgWFXtBA5I3A52VZTLwgu+9nQQjoZ715otB45ttpfXEN1R/uu8H4XI8XYS/mA5a5z8PuPsC7adROIc2g";
    std::string key = "c31b1290659cc5d08a00fdd556dc330545486545a3251c3681254db33b9e689179ddbfbaf6a13fafe127097ff1efb646911535d1384239178589065c5bc8c024";
    std::string submitaccount = "...";
    std::string worker_id = "...";

    nlohmann::json payload = {
        {"hash_to_verify", hashed_data},
        {"key", key},
        {"account", submitaccount},
        {"attempts", "123456"},
        {"hashes_per_second", "1234"},
        {"worker", worker_id}
    };

    std::cout << "Payload: " << payload.dump(4) << std::endl;

    std::string response = HttpPostRequest("http://xenblocks.io/verify", payload, 10); // 10 seconds timeout
    
    std::cout << "Server Response: " << response << std::endl;

    try {
        auto jsonResponse = nlohmann::json::parse(response);
        std::cout << "Server Response: " << jsonResponse.dump(4) << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Failed to parse JSON response: " << e.what() << std::endl;
    }

    return 0;
}
