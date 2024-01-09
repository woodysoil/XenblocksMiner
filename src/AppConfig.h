// AppConfig.h

#ifndef APP_CONFIG_H
#define APP_CONFIG_H

#include "ConfigManager.h"
#include <string>

class AppConfig {
public:
    AppConfig(const std::string& filename) : configFileName(filename) {}

    void load();
    const std::string& getAccountAddress() const { return accountAddress; };
    const int getDevfeePermillage() const { return devfeePermillage; };

private:
    bool isValidEIP55Address(const std::string& address);
    bool isValidDevfee(int devfee);
    std::string promptForEIP55Address();
    int promptForDevfeePermillage();

    std::string configFileName;
    std::string accountAddress;
    int devfeePermillage;
};

#endif // APP_CONFIG_H
