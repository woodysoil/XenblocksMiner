cmake_minimum_required(VERSION 3.18)
project(submitpow)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(OpenSSL REQUIRED)
if(NOT OPENSSL_FOUND)
    message(FATAL_ERROR "OpenSSL not found. Please install OpenSSL.")
endif()
find_package(CURL REQUIRED)
if(NOT CURL_FOUND)
    message(FATAL_ERROR "libcurl not found. Please install libcurl.")
endif()
find_package(nlohmann_json REQUIRED)
if(NOT nlohmann_json_FOUND)
    message(FATAL_ERROR "nlohmann_json library not found. Please install nlohmann_json.")
endif()

find_package(PkgConfig REQUIRED)
pkg_check_modules(ARGON2 REQUIRED libargon2)

if(NOT ARGON2_FOUND)
    message(FATAL_ERROR "Libargon2 not found")
endif()

include_directories(${ARGON2_INCLUDE_DIRS})
link_directories(${ARGON2_LIBRARY_DIRS})

add_executable(${PROJECT_NAME} main.cpp MerkleTree.cpp HttpClient.cpp HttpResponse.cpp Argon2idHasher.cpp SHA256Hasher.cpp)

target_link_libraries(${PROJECT_NAME} OpenSSL::SSL OpenSSL::Crypto CURL::libcurl nlohmann_json::nlohmann_json ${ARGON2_LIBRARIES})
