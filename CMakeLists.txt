cmake_minimum_required(VERSION 3.18)

project(xenblocksMiner LANGUAGES CUDA CXX VERSION 1.0)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 35 37 50 52 61 70 75 80 86 89 90 CACHE STRING "CUDA architectures")

if(WIN32)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --allow-unsupported-compiler")
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

find_package(CUDAToolkit REQUIRED)

find_package(unofficial-argon2 CONFIG REQUIRED)

find_package(cpr CONFIG REQUIRED)
if(NOT cpr_FOUND)
    message(FATAL_ERROR "libcpr not found. Please install libcpr.")
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

find_package(Boost REQUIRED COMPONENTS program_options)
if(NOT Boost_FOUND)
    message(FATAL_ERROR "Boost.Program_options not found. Please install Boost.Program_options.")
endif()

find_package(unofficial-secp256k1 CONFIG REQUIRED)
if(NOT unofficial-secp256k1_FOUND)
    message(FATAL_ERROR "Libsecp256k1 not found. Please install Libsecp256k1.")
endif()

find_package(Crow CONFIG REQUIRED)
if(NOT Crow_FOUND)
    message(FATAL_ERROR "Libcrow not found. Please install Libcrow.")
endif()

list(APPEND LINK_LIBRARIES CUDA::nvml unofficial::argon2::libargon2 cryptopp::cryptopp cpr::cpr nlohmann_json::nlohmann_json OpenSSL::SSL OpenSSL::Crypto Boost::program_options unofficial::secp256k1 unofficial::secp256k1_precomputed Crow::Crow)

set(SOURCE_FILES src/main.cpp src/AppConfig.cpp src/argon2params.cpp src/blake2b.cpp src/ConfigManager.cpp src/CudaDevice.cpp src/kernelrunner.cu src/MineUnit.cpp src/MiningCommon.cpp src/Logger.cpp src/MerkleTree.cpp src/SHA256Hasher.cpp src/EthereumAddressValidator.cpp src/Argon2idHasher.cpp src/HttpClient.cpp src/HttpResponse.cpp src/PowSubmitter.cpp src/EthereumSignatureValidator.cpp src/MachineIDGetter.cpp)
if(WIN32)
    add_executable(${PROJECT_NAME} ${SOURCE_FILES} res/version.rc)
else()
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static-libgcc -static-libstdc++")
    add_executable(${PROJECT_NAME} ${SOURCE_FILES})
endif()

if(WIN32)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${LINK_LIBRARIES} Iphlpapi)
elseif(UNIX AND NOT APPLE)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${LINK_LIBRARIES})
endif()

target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>
)
