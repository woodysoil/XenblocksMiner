cmake_minimum_required(VERSION 3.18)

project(xenblocksMiner LANGUAGES CUDA CXX VERSION 1.0)

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 35 37 50 52 61 70 75 80 86 89 90)

find_package(unofficial-argon2 CONFIG REQUIRED)
if(NOT unofficial-argon2_FOUND)
    message(FATAL_ERROR "Libargon2 not found. Please install Libargon2.")
endif()

find_package(CURL CONFIG REQUIRED)
if(NOT CURL_FOUND)
    message(FATAL_ERROR "libcurl not found. Please install libcurl.")
endif()

find_package(nlohmann_json CONFIG REQUIRED)
if(NOT nlohmann_json_FOUND)
    message(FATAL_ERROR "nlohmann_json library not found. Please install nlohmann_json.")
endif()

find_package(cryptopp CONFIG REQUIRED)
if (NOT cryptopp_FOUND)
    message(FATAL_ERROR "Crypto++ library not found. Please install Crypto++.")
endif()

find_package(OpenSSL REQUIRED)
if(NOT OPENSSL_FOUND)
    message(FATAL_ERROR "OpenSSL not found. Please install OpenSSL.")
endif()

list(APPEND LINK_LIBRARIES unofficial::argon2::libargon2 cryptopp::cryptopp CURL::libcurl nlohmann_json::nlohmann_json OpenSSL::SSL OpenSSL::Crypto)

set(SOURCE_FILES src/main.cpp src/AppConfig.cpp src/argon2params.cpp src/blake2b.cpp src/ConfigManager.cpp src/CudaDevice.cpp src/kernelrunner.cu src/MineUnit.cpp src/MiningCommon.cpp src/Logger.cpp src/MerkleTree.cpp src/SHA256Hasher.cpp src/EthereumAddressValidator.cpp src/Argon2idHasher.cpp src/HttpClient.cpp src/HttpResponse.cpp src/PowSubmitter.cpp)
if(WIN32)
    add_executable(${PROJECT_NAME} ${SOURCE_FILES} res/version.rc)
else()
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static-libgcc -static-libstdc++")
    add_executable(${PROJECT_NAME} ${SOURCE_FILES})
endif()

target_link_libraries(${PROJECT_NAME} PRIVATE ${LINK_LIBRARIES})

target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O2>
    $<$<COMPILE_LANGUAGE:CUDA>:-O2>
)

set_target_properties(${PROJECT_NAME} PROPERTIES 
    COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>"
)