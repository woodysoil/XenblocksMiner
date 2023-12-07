// ConfigManager.h

#ifndef CONFIG_MANAGER_H
#define CONFIG_MANAGER_H

#include <iostream>
#include <fstream>
#include <unordered_map>
#include <string>
#include <algorithm>

class ConfigManager {
public:
    ConfigManager(const std::string& filename);

    void loadConfig();
    std::string getConfigValue(const std::string& key);
    void setConfigValue(const std::string& key, const std::string& value);
    void saveConfig();

private:
    std::string trim(const std::string& str);

    std::string filename_;
    std::unordered_map<std::string, std::string> config_;
};

#endif // CONFIG_MANAGER_H
