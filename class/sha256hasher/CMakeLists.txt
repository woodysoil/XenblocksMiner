cmake_minimum_required(VERSION 3.18)
project(sha256hasher)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(OpenSSL REQUIRED)
if(NOT OPENSSL_FOUND)
    message(FATAL_ERROR "OpenSSL not found. Please install OpenSSL.")
endif()

add_executable(${PROJECT_NAME} main.cpp SHA256Hasher.cpp)

target_link_libraries(${PROJECT_NAME} OpenSSL::SSL OpenSSL::Crypto)
