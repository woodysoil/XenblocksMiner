// AppConfig.h

#ifndef APP_CONFIG_H
#define APP_CONFIG_H

#include "ConfigManager.h"
#include <string>

class AppConfig {
public:
    AppConfig(const std::string& filename) : configFileName(filename) {}

    void load();
    void tryLoad();
    const std::string& getAccountAddress() const { return accountAddress; };
    const std::string& getEcoDevAddr() const { return ecoDevAddr; };
    const int getDevfeePermillage() const { return devfeePermillage; };

    void save();
    void setAccountAddress(const std::string& address) { accountAddress = address; };
    void setEcoDevAddr(const std::string& address) { ecoDevAddr = address; };
    void setDevfeePermillage(int devfee) { devfeePermillage = devfee; };

private:
    bool isValidEIP55Address(const std::string& address);
    bool isValidDevfee(int devfee);
    std::string promptForEIP55Address();
    int promptForDevfeePermillage();

    std::string configFileName;
    std::string accountAddress;
    std::string ecoDevAddr;
    int devfeePermillage = 0;
};

#endif // APP_CONFIG_H
