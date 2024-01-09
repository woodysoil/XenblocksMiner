#include "HttpResponse.h"

HttpResponse::HttpResponse(int code, const std::string& body, const std::map<std::string, std::string>& headers)
    : statusCode(code), responseBody(body), responseHeaders(headers) {}

int HttpResponse::GetStatusCode() const {
    return statusCode;
}

const std::string& HttpResponse::GetBody() const {
    return responseBody;
}

const std::map<std::string, std::string>& HttpResponse::GetHeaders() const {
    return responseHeaders;
}
