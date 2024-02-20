#ifndef ETHEREUMSIGNATUREVALIDATOR_H
#define ETHEREUMSIGNATUREVALIDATOR_H

#include <string>
#include <vector>

class EthereumSignatureValidator {
public:
    static bool verifyMessage(const std::string& message, const std::string& signatureHex, const std::string& expectedAddress);
};

#endif // ETHEREUMSIGNATUREVALIDATOR_H
