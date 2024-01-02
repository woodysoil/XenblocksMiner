#pragma once

#include <iostream>
#include <string>
#include <random>
#include <sstream>
#include <chrono>

class RandomHexKeyGenerator {
public:
    RandomHexKeyGenerator(const std::string& initial_prefix = "", size_t key_length = 64)
        : prefix(initial_prefix), total_length(key_length) {
            std::random_device rd;
            auto seed = rd() ^ static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
            generator.seed(seed);
        }

    void setPrefix(const std::string& new_prefix) {
        prefix = new_prefix;
    }

    std::string nextRandomKey() {
        const std::string hex_chars = "0123456789abcdefABCDEF";
        std::uniform_int_distribution<size_t> distribution(0, hex_chars.size() - 1);

        std::stringstream ss;

        if (prefix.length() >= total_length) {
            std::cout << "Wraning: Prefix is longer than total length. Returning prefix." << std::endl;
            ss << prefix.substr(0, total_length);
        }
        else {
            ss << prefix;
            size_t remaining_length = total_length - prefix.length();

            for (size_t i = 0; i < remaining_length; ++i) {
                ss << hex_chars[distribution(generator)];
            }
        }

        return ss.str();
    }

private:
    std::string prefix;
    size_t total_length;
    std::mt19937 generator;
};