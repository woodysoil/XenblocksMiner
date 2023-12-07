#include <iostream>
#include <fstream>
#include <unordered_map>
#include <string>
#include <algorithm>


// Class to manage configuration settings
class ConfigManager {
public:
    // Constructor taking the filename of the config file
    ConfigManager(const std::string& filename) : filename_(filename) {}

    // Load the configuration from a file
    void loadConfig() {
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

    // Get a configuration value by key
    std::string getConfigValue(const std::string& key) {
        return config_.find(key) != config_.end() ? config_[key] : "";
    }

    // Set a configuration value by key
    void setConfigValue(const std::string& key, const std::string& value) {
        config_[key] = value;
    }

    // Save the current configuration to a file
    void saveConfig() {
        std::ofstream file(filename_);
        for (const auto& pair : config_) {
            file << pair.first << "=" << pair.second << std::endl;
        }
        file.close();
    }

private:
// Function to trim whitespace from both ends of a string
std::string trim(const std::string& str) {
    const auto strBegin = str.find_first_not_of(" \t");
    if (strBegin == std::string::npos)
        return ""; // no content

    const auto strEnd = str.find_last_not_of(" \t");
    const auto strRange = strEnd - strBegin + 1;

    return str.substr(strBegin, strRange);
}

private:
    std::string filename_;  // Filename of the config file
    std::unordered_map<std::string, std::string> config_;  // Map to store key-value pairs
};

int main() {
    ConfigManager manager("config.txt");

    // Load configuration
    manager.loadConfig();

    // Demonstrate reading configuration values
    std::cout << "Account: " << manager.getConfigValue("account") << std::endl;
    std::cout << "Last Block URL: " << manager.getConfigValue("last_block_url") << std::endl;
    std::cout << "Developer Fee Enabled: " << manager.getConfigValue("dev_fee_on") << std::endl;
    std::cout << "GPU: " << manager.getConfigValue("gpu") << std::endl;

    // Modify configuration
    manager.setConfigValue("account", "0x24691e54afafe2416a8252097c9ca67557271475");
    manager.setConfigValue("last_block_url", "http://xenminer.mooo.com:4445/getblocks/lastblock");
    manager.setConfigValue("dev_fee_on", "false");
    manager.setConfigValue("gpu", "2");

    // Save configuration
    manager.saveConfig();

    // Demonstrate reading configuration values
    std::cout << "Account: " << manager.getConfigValue("account") << std::endl;
    std::cout << "Last Block URL: " << manager.getConfigValue("last_block_url") << std::endl;
    std::cout << "Developer Fee Enabled: " << manager.getConfigValue("dev_fee_on") << std::endl;
    std::cout << "GPU: " << manager.getConfigValue("gpu") << std::endl;

    return 0;
}
