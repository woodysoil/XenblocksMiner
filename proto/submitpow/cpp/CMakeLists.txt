cmake_minimum_required(VERSION 3.18)
project(submitpow)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(CURL REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(ARGON2 REQUIRED libargon2)

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME} CURL::libcurl nlohmann_json::nlohmann_json ${ARGON2_LIBRARIES} OpenSSL::SSL OpenSSL::Crypto)
