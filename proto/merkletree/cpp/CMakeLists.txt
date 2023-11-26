cmake_minimum_required(VERSION 3.18)

project(merkleTreeBuilder)

find_package(OpenSSL REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME} OpenSSL::SSL OpenSSL::Crypto)
