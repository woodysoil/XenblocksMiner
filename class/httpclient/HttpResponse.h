#ifndef HTTPRESPONSE_H
#define HTTPRESPONSE_H

#include <string>
#include <map>

class HttpResponse {
public:
    HttpResponse(int code, const std::string& body, const std::map<std::string, std::string>& headers);

    int GetStatusCode() const;
    const std::string& GetBody() const;
    const std::map<std::string, std::string>& GetHeaders() const;

private:
    int statusCode;
    std::string responseBody;
    std::map<std::string, std::string> responseHeaders;
};

#endif // HTTPRESPONSE_H
