#include "DifficultyManager.h"

#include <iostream>
#include <thread>
#include <chrono>
#include <stdexcept>
#include <nlohmann/json.hpp>
#include "HttpClient.h"
#include "MiningCommon.h"

std::string getDifficulty()
{
    HttpClient httpClient;

    try
    {
        HttpResponse response = httpClient.HttpGet(globalRpcLink+"/difficulty", 5000);
        if (response.GetStatusCode() != 200)
        {
            throw std::runtime_error("Failed to get the difficulty: HTTP status code " + std::to_string(response.GetStatusCode()));
        }

        auto json_response = nlohmann::json::parse(response.GetBody());
        return json_response["difficulty"].get<std::string>();
    }
    catch (const nlohmann::json::parse_error &e)
    {
        throw std::runtime_error("JSON parsing error: " + std::string(e.what()));
    }
    catch (const std::exception &e)
    {
        throw std::runtime_error("Error: " + std::string(e.what()));
    }
}

void updateDifficulty()
{
    try
    {
        std::string difficultyStr = getDifficulty();
        int newDifficulty = std::stoi(difficultyStr);

        std::lock_guard<std::mutex> lock(mtx);
        if (globalDifficulty != newDifficulty)
        {
            globalDifficulty = newDifficulty;
            std::cout << "Updated difficulty to " << globalDifficulty << std::endl;
        }
    }
    catch (const std::exception &e)
    {
    }
}

void updateDifficultyPeriodically()
{
    while (running)
    {
        updateDifficulty();
        std::this_thread::sleep_for(std::chrono::seconds(10));
    }
}
