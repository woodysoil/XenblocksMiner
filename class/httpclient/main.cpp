#include <iostream>
#include <nlohmann/json.hpp>
#include "HttpClient.h"

std::string getDifficulty() {
    HttpClient httpClient;

    try {
        HttpResponse response = httpClient.HttpGet("http://xenblocks.io/difficulty", 10); // 10 seconds timeout
        if (response.GetStatusCode() != 200) {
            throw std::runtime_error("Failed to get the difficulty: HTTP status code " + std::to_string(response.GetStatusCode()));
        }

        auto json_response = nlohmann::json::parse(response.GetBody());
        return json_response["difficulty"].get<std::string>();
    } catch (const nlohmann::json::parse_error& e) {
        throw std::runtime_error("JSON parsing error: " + std::string(e.what()));
    } catch (const std::exception& e) {
        throw std::runtime_error("Error: " + std::string(e.what()));
    }
}

int main() {
    HttpClient httpClient;

    std::string hashed_data = "$argon2id$v=19$m=95400,t=1,p=1$WEVOMTAwODIwMjJYRU4$StL5GRjgWFXtBA5I3A52VZTLwgu+9nQQjoZ715otB45ttpfXEN1R/uu8H4XI8XYS/mA5a5z8PuPsC7adROIc2g";
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

    // Use HttpClient to send POST request
    HttpResponse response = httpClient.HttpPost("http://xenblocks.io/verify", payload, 10); // 10 seconds timeout
    
    std::cout << "Server Response: " << response.GetBody() << std::endl;
    std::cout << "Status Code: " << response.GetStatusCode() << std::endl;

    try {
        auto jsonResponse = nlohmann::json::parse(response.GetBody());
        std::cout << "Parsed JSON Response: " << jsonResponse.dump(4) << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Failed to parse JSON response: " << e.what() << std::endl;
    }

    try {
        std::cout << "Difficulty: " << getDifficulty() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}
