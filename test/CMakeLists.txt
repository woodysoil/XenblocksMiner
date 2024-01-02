cmake_minimum_required(VERSION 3.18)

project(xenblocksMiner LANGUAGES CUDA CXX VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_FLAGS_RELEASE "-O2")
set(CMAKE_C_FLAGS_RELEASE "-O2")

set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 35 37 50 52 61 70 75 80 86)# 89 90 
# nvcc 11.8 3.5-9.0
# nvcc 10.2 3.0-7.x

find_package(PkgConfig REQUIRED)
pkg_check_modules(ARGON2 REQUIRED libargon2)
if(NOT ARGON2_FOUND)
    message(FATAL_ERROR "Libargon2 not found")
endif()

find_library(CRYPTOPP_LIBRARY cryptopp)
if (NOT CRYPTOPP_LIBRARY)
    message(FATAL_ERROR "Crypto++ library not found")
endif()

find_package(CURL REQUIRED)
if(NOT CURL_FOUND)
    message(FATAL_ERROR "libcurl not found. Please install libcurl.")
endif()

find_package(nlohmann_json REQUIRED)
if(NOT nlohmann_json_FOUND)
    message(FATAL_ERROR "nlohmann_json library not found. Please install nlohmann_json.")
endif()

find_package(OpenSSL REQUIRED)
if(NOT OPENSSL_FOUND)
    message(FATAL_ERROR "OpenSSL not found. Please install OpenSSL.")
endif()

include_directories(${ARGON2_INCLUDE_DIRS})
link_directories(${ARGON2_LIBRARY_DIRS})

add_executable(${PROJECT_NAME} main.cpp AppConfig.cpp argon2params.cpp blake2b.cpp ConfigManager.cpp CudaDevice.cpp kernelrunner.cu MineUnit.cpp MiningCommon.cpp Logger.cpp MerkleTree.cpp SHA256Hasher.cpp EthereumAddressValidator.cpp Argon2idHasher.cpp HttpClient.cpp HttpResponse.cpp PowSubmitter.cpp)

target_link_libraries(${PROJECT_NAME} PRIVATE ${ARGON2_LIBRARIES} ${CRYPTOPP_LIBRARY} CURL::libcurl nlohmann_json::nlohmann_json OpenSSL::SSL OpenSSL::Crypto)

set_target_properties(${PROJECT_NAME} PROPERTIES 
    LINK_FLAGS "-static-libgcc -static-libstdc++"
    COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>"
)