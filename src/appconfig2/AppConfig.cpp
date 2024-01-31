#include "AppConfig.h"
#include "MiningCommon.h"
#include "EthereumAddressValidator.h"

void AppConfig::load()
{
    std::ifstream configFile(configFileName);
    ConfigManager configManager(configFileName);
    if (configFile.good()) {
        configManager.loadConfig();
        accountAddress = configManager.getConfigValue("account_address");
        if (!isValidEIP55Address(accountAddress)) {
            std::cout << RED << "The account address in the configuration file is invalid." << RESET << std::endl;
            std::cout << "Please enter a valid EIP-55 account address (Ethereum address) again." << std::endl;
            accountAddress = promptForEIP55Address();
        }

        try {
            devfeePermillage = std::stoi(configManager.getConfigValue("devfee_permillage"));
            if (!isValidDevfee(devfeePermillage)) {
                throw std::invalid_argument("Invalid devfee permillage.");
            }
        }
        catch (const std::invalid_argument& e) {
            std::cout << RED << "The devfee permillage in the configuration file is invalid. " << RESET << std::endl;
            std::cout << "Please enter a valid devfee per thousand (range 0 - 1000) again." << std::endl;

            devfeePermillage = promptForDevfeePermillage();
        }
    }
    else {
        std::cout << YELLOW << "Welcome! It looks like this is your first time running the application. Let's set up the necessary configurations." << RESET << std::endl << std::endl;
        accountAddress = promptForEIP55Address();
       devfeePermillage = promptForDevfeePermillage();
        std::cout << std::endl;
        std::cout << GREEN << "All set! Your configurations are saved and the application is ready to use." << RESET << std::endl;
        std::cout << std::endl;
    }

    configManager.setConfigValue("account_address", accountAddress);
    configManager.setConfigValue("devfee_permillage", std::to_string(devfeePermillage));
    configManager.saveConfig();
}

bool AppConfig::isValidEIP55Address(const std::string& address)
{
    EthereumAddressValidator validator;
    if (validator.isValid(address)) {
        return true;
    } else {
        std::cout << "Invalid Ethereum address" << std::endl;
        return false;
    }
}

bool AppConfig::isValidDevfee(int devfee)
{
    return devfee >= 0 && devfee <= 1000;
}

std::string AppConfig::promptForEIP55Address()
{
    std::string address;
    do {
        std::cout << "Enter valid EIP-55 account address: ";
        std::cin >> address;
    } while (!isValidEIP55Address(address));
    return address;
}

int AppConfig::promptForDevfeePermillage()
{
    int devfee;
    do {
        std::cout << "Enter devfee per thousand (0-1000): ";
        std::cin >> devfee;
        if (!std::cin) {
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        }
    } while (!isValidDevfee(devfee));
    return devfee;
}
