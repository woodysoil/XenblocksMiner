#pragma once

#include <iostream>
#include <string>
#include <random>
#include <chrono>
#include <algorithm>
#include <cctype>

class RandomHexKeyGenerator {
public:
    RandomHexKeyGenerator(const std::string& initial_prefix = "", size_t key_length = 64)
        : total_length(key_length), distribution(0, 15) {
            setPrefix(initial_prefix);
            std::random_device rd;
            auto seed = rd() ^ static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
            generator.seed(seed);
        }

    void setPrefix(const std::string& new_prefix) {
        prefix = new_prefix;
        std::transform(prefix.begin(), prefix.end(), prefix.begin(),
                       [](unsigned char c){ return std::tolower(c); });
    }

    std::string nextRandomKey() {
        if (prefix.length() >= total_length) {
            std::cout << "Warning: Prefix is longer than total length. Returning prefix." << std::endl;
            return prefix.substr(0, total_length);
        }

        std::string key;
        key.reserve(total_length);
        key.append(prefix);
        while (key.length() < total_length) {
            key.push_back(kHexChars[distribution(generator)]);
        }
        return key;
    }

private:
    inline static constexpr char kHexChars[] = "0123456789abcdef";
    std::string prefix;
    size_t total_length;
    std::mt19937 generator;
    std::uniform_int_distribution<size_t> distribution;
};
