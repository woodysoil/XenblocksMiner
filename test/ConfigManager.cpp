// ConfigManager.cpp

#include "ConfigManager.h"

ConfigManager::ConfigManager(const std::string& filename) : filename_(filename) {}

void ConfigManager::loadConfig() {
    std::ifstream file(filename_);
    std::string line;
    while (std::getline(file, line)) {
        auto delimiterPos = line.find('=');
        if (delimiterPos != std::string::npos) {
            auto key = trim(line.substr(0, delimiterPos));
            auto value = trim(line.substr(delimiterPos + 1));
            config_[key] = value;
        }
    }
    file.close();
}

std::string ConfigManager::getConfigValue(const std::string& key) {
    return config_.find(key) != config_.end() ? config_[key] : "";
}

void ConfigManager::setConfigValue(const std::string& key, const std::string& value) {
    config_[key] = value;
}

void ConfigManager::saveConfig() {
    std::ofstream file(filename_);
    for (const auto& pair : config_) {
        file << pair.first << "=" << pair.second << std::endl;
    }
    file.close();
}

std::string ConfigManager::trim(const std::string& str) {
    const auto strBegin = str.find_first_not_of(" \t");
    if (strBegin == std::string::npos)
        return "";
    
    const auto strEnd = str.find_last_not_of(" \t");
    const auto strRange = strEnd - strBegin + 1;

    return str.substr(strBegin, strRange);
}
