#include <iostream>
#include <curl/curl.h>

size_t callback(const char* in, size_t size, size_t num, std::string* out) {
    const size_t totalBytes(size * num);
    out->append(in, totalBytes);
    return totalBytes;
}

int main() {
    CURL* curl = curl_easy_init();

    if (curl) {
        curl_easy_setopt(curl, CURLOPT_URL, "http://localhost:2357/");
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, callback);

        std::string response_string;
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_string);

        CURLcode res = curl_easy_perform(curl);
        if (res != CURLE_OK) {
            std::cerr << "curl_easy_perform() failed: " << curl_easy_strerror(res) << std::endl;
        } else {
            std::cout << "Response: " << response_string << std::endl;
        }

        curl_easy_cleanup(curl);
    }

    return 0;
}
