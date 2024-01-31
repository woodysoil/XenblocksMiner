#ifndef HTTPCLIENT_H
#define HTTPCLIENT_H

#include <string>
#include <nlohmann/json.hpp>
#include <curl/curl.h>
#include <iostream>
#include <map>
#include "HttpResponse.h"

class HttpClient {
public:
    HttpResponse HttpGet(const std::string& url, long timeout);
    HttpResponse HttpPost(const std::string& url, const nlohmann::json& payload, long timeout);

private:
    static size_t WriteCallback(void *contents, size_t size, size_t nmemb, std::string *s);
    static size_t HeaderCallback(char *buffer, size_t size, size_t nitems, std::map<std::string, std::string> *headers);
};

#endif // HTTPCLIENT_H
