#include "HttpClient.h"
#include <cpr/cpr.h>
#include <map>
#include <nlohmann/json.hpp>
#include <string>

#include <iostream>

HttpResponse HttpClient::HttpGet(const std::string &url, long timeout) {
    auto response = cpr::Get(cpr::Url{url}, cpr::Timeout{timeout});

    std::map<std::string, std::string> headers;
    for (const auto &header : response.header) {
        headers[header.first] = header.second;
    }

    // if (response.status_code == 0) {
    //     std::cout << "HTTP GET Request to URL: " << url << std::endl;
    //     std::cout << "Status Code: " << response.status_code << std::endl;
    //     if (response.error) {
    //         std::cout << "cpr::Error message: " << response.error.message
    //                   << std::endl;
    //     }

    //     std::cout << "Response Headers:" << std::endl;
    //     for (const auto &header : headers) {
    //         std::cout << header.first << ": " << header.second << std::endl;
    //     }

    //     std::cout << "Response Body Preview: " << response.text << std::endl;
    // }

    return HttpResponse(response.status_code, response.text, headers);
}

HttpResponse HttpClient::HttpPost(const std::string &url,
                                  const nlohmann::json &payload, long timeout) {
    auto response = cpr::Post(cpr::Url{url}, cpr::Body{payload.dump()},
                              cpr::Header{{"Content-Type", "application/json"}},
                              cpr::Timeout{timeout});

    std::map<std::string, std::string> headers;
    for (const auto &header : response.header) {
        headers[header.first] = header.second;
    }

    // std::cout << "HTTP POST Request to URL: " << url << std::endl;
    // std::cout << "Payload: " << payload.dump() << std::endl;
    // std::cout << "Status Code: " << response.status_code << std::endl;
    // if (response.error) {
    //     std::cout << "cpr::Error message: " << response.error.message
    //               << std::endl;
    // }

    // std::cout << "Response Headers:" << std::endl;
    // for (const auto &header : headers) {
    //     std::cout << header.first << ": " << header.second << std::endl;
    // }

    // std::cout << "Response Body Preview: " << response.text << std::endl;

    return HttpResponse(response.status_code, response.text, headers);
}