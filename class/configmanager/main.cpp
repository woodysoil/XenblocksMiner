// main.cpp

#include "ConfigManager.h"

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
