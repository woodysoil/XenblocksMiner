#include "HttpClient.h"

// Callback function writes data to a std::string, and will be called by libcurl as data arrives
size_t HttpClient::WriteCallback(void *contents, size_t size, size_t nmemb, std::string *s) {
    size_t newLength = size * nmemb;
    try {
        s->append((char*)contents, newLength);
    } catch(std::bad_alloc &e) {
        // Handle memory problem
        return 0;
    }
    return newLength;
}

// Header callback function to capture response headers
size_t HttpClient::HeaderCallback(char *buffer, size_t size, size_t nitems, std::map<std::string, std::string> *headers) {
    std::string header(buffer, nitems * size);
    size_t separator = header.find(':');
    if (separator != std::string::npos) {
        std::string key = header.substr(0, separator);
        // Trim leading and trailing whitespace from the value
        size_t valueStart = header.find_first_not_of(" \t", separator + 1);
        size_t valueEnd = header.find_last_not_of("\r\n");
        std::string value = header.substr(valueStart, (valueEnd - valueStart + 1));

        (*headers)[key] = value;
    }
    return nitems * size;
}


// HTTP GET request function
HttpResponse HttpClient::HttpGet(const std::string& url, long timeout) {
    CURL *curl;
    CURLcode res;
    long response_code = 0;
    std::string readBuffer;
    std::map<std::string, std::string> headers;

    curl = curl_easy_init();
    if(curl) {
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, HeaderCallback);
        curl_easy_setopt(curl, CURLOPT_HEADERDATA, &headers);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, timeout);

        res = curl_easy_perform(curl);
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);
        if (res != CURLE_OK) {
            // std::cerr << "curl_easy_perform() failed: " << curl_easy_strerror(res) << std::endl;
        }
        curl_easy_cleanup(curl);
    }
    return HttpResponse(response_code, readBuffer, headers);
}

// HTTP POST request function
HttpResponse HttpClient::HttpPost(const std::string& url, const nlohmann::json& payload, long timeout) {
    std::map<std::string, std::string> responseHeaders;
    CURL *curl;
    CURLcode res;
    std::string readBuffer;
    struct curl_slist *headers = nullptr;
    long response_code = 0;
    curl = curl_easy_init();
    if(curl) {
        headers = curl_slist_append(headers, "Content-Type: application/json");
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        std::string json_payload = payload.dump();
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_payload.c_str());
        curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, json_payload.size());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, timeout);
        // curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L); // detail output

        res = curl_easy_perform(curl);
        if (res != CURLE_OK) {
            // std::cerr << "curl_easy_perform() failed: " << curl_easy_strerror(res) << std::endl;
        }
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);

        curl_slist_free_all(headers);  // free the header list
        curl_easy_cleanup(curl);
    }
    return HttpResponse(response_code, readBuffer, responseHeaders);
}