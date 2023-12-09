cmake_minimum_required(VERSION 3.18)
project(httpClient)

set(CMAKE_CXX_STANDARD 17)

find_package(CURL REQUIRED)
if(NOT CURL_FOUND)
    message(FATAL_ERROR "libcurl not found. Please install libcurl.")
endif()

find_package(nlohmann_json REQUIRED)
if(NOT nlohmann_json_FOUND)
    message(FATAL_ERROR "nlohmann_json library not found. Please install nlohmann_json.")
endif()

add_executable(${PROJECT_NAME} main.cpp HttpClient.cpp HttpResponse.cpp)


target_link_libraries(${PROJECT_NAME} CURL::libcurl nlohmann_json::nlohmann_json)