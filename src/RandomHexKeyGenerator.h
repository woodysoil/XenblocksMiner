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
        : total_length(key_length) {
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
            std::uint32_t random_bits = generator();
            for (int nibble = 0; nibble < 8 && key.length() < total_length; ++nibble) {
                key.push_back(kHexChars[random_bits & 0x0f]);
                random_bits >>= 4;
            }
        }
        return key;
    }

private:
    inline static constexpr char kHexChars[] = "0123456789abcdef";
    std::string prefix;
    size_t total_length;
    std::mt19937 generator;
};
