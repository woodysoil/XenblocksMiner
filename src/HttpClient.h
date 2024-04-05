#ifndef HTTPCLIENT_H
#define HTTPCLIENT_H

#include <string>
#include <nlohmann/json.hpp>
#include "HttpResponse.h"

class HttpClient {
public:
    HttpResponse HttpGet(const std::string& url, long timeout = 60000);
    HttpResponse HttpPost(const std::string& url, const nlohmann::json& payload, long timeout = 60000);
};

#endif // HTTPCLIENT_H
