cmake_minimum_required(VERSION 3.18)

option(XENBLOCKS_BUILD_MINER "Build the full CUDA miner" ON)
option(XENBLOCKS_BUILD_HASHAPI_CLI "Build the standalone Hash API CLI" OFF)
option(XENBLOCKS_HASHAPI_STUB_BACKEND "Build hashapi-cli with a deterministic stub backend for CLI smoke tests" OFF)

if(XENBLOCKS_BUILD_MINER)
    project(xenblocksMiner LANGUAGES CUDA CXX VERSION 1.0)
else()
    project(xenblocksMiner LANGUAGES CXX VERSION 1.0)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(XENBLOCKS_BUILD_MINER)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_ARCHITECTURES 50 52 61 70 75 80 86 89 90 CACHE STRING "CUDA architectures")
endif()

if(WIN32)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

if(XENBLOCKS_BUILD_MINER)
    find_package(CUDAToolkit REQUIRED)
    find_package(cpr CONFIG REQUIRED)
    if(NOT cpr_FOUND)
        message(FATAL_ERROR "libcpr not found. Please install libcpr.")
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

    find_package(PahoMqttCpp CONFIG REQUIRED)
    if(NOT PahoMqttCpp_FOUND)
        message(FATAL_ERROR "PahoMqttCpp not found. Please install PahoMqttCpp.")
    endif()
    if(TARGET PahoMqttCpp::paho-mqttpp3-static)
        set(XENBLOCKS_PAHO_MQTT_CPP_TARGET PahoMqttCpp::paho-mqttpp3-static)
    elseif(TARGET PahoMqttCpp::paho-mqttpp3-shared)
        set(XENBLOCKS_PAHO_MQTT_CPP_TARGET PahoMqttCpp::paho-mqttpp3-shared)
        set(XENBLOCKS_PAHO_MQTT_CPP_SHARED ON)
    elseif(TARGET PahoMqttCpp::paho-mqttpp3)
        set(XENBLOCKS_PAHO_MQTT_CPP_TARGET PahoMqttCpp::paho-mqttpp3)
    else()
        message(FATAL_ERROR "PahoMqttCpp target not found after find_package(PahoMqttCpp).")
    endif()
endif()

find_package(unofficial-argon2 CONFIG QUIET)
if(NOT unofficial-argon2_FOUND)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(ARGON2 QUIET libargon2)
    endif()
endif()

find_package(nlohmann_json CONFIG QUIET)

if(XENBLOCKS_BUILD_MINER)
    if(NOT unofficial-argon2_FOUND)
        message(FATAL_ERROR "libargon2 not found. Please install argon2.")
    endif()
    if(NOT nlohmann_json_FOUND)
        message(FATAL_ERROR "nlohmann_json library not found. Please install nlohmann_json.")
    endif()
    list(APPEND LINK_LIBRARIES CUDA::nvml unofficial::argon2::libargon2 cryptopp::cryptopp cpr::cpr nlohmann_json::nlohmann_json OpenSSL::SSL OpenSSL::Crypto Boost::program_options unofficial::secp256k1 unofficial::secp256k1_precomputed Crow::Crow ${XENBLOCKS_PAHO_MQTT_CPP_TARGET})
endif()

set(SOURCE_FILES
    src/main.cpp
    src/ProcessMonitor.cpp
    src/DifficultyManager.cpp
    src/StatReporter.cpp
    src/LocalServer.cpp
    src/BlockSubmitter.cpp
    src/AppConfig.cpp
    src/argon2params.cpp
    src/blake2b.cpp
    src/ConfigManager.cpp
    src/CudaDevice.cpp
    src/CudaBackend.cpp
    src/kernelrunner.cu
    src/MineUnit.cpp
    src/MiningCommon.cpp
    src/Logger.cpp
    src/MerkleTree.cpp
    src/SHA256Hasher.cpp
    src/EthereumAddressValidator.cpp
    src/Argon2idHasher.cpp
    src/HttpClient.cpp
    src/HttpResponse.cpp
    src/PowSubmitter.cpp
    src/EthereumSignatureValidator.cpp
    src/MachineIDGetter.cpp
    src/MiningCoordinator.cpp
    src/MqttClient.cpp
    src/WorkerReporter.cpp
    src/LeaseManager.cpp
    src/PlatformManager.cpp
    src/hashapi/CpuHashBackend.cpp
    src/hashapi/CudaHashBackend.cpp
    src/hashapi/HashApiCli.cpp
    src/hashapi/HashApiEncoding.cpp
    src/hashapi/HashApiJson.cpp
    src/hashapi/HashApiMatching.cpp
    src/hashapi/HashApiValidation.cpp
)
set(HASHAPI_SOURCE_FILES
    src/hashapi/HashApiCli.cpp
    src/hashapi/HashApiEncoding.cpp
    src/hashapi/HashApiJson.cpp
    src/hashapi/HashApiMatching.cpp
    src/hashapi/HashApiValidation.cpp
)

if(XENBLOCKS_HASHAPI_STUB_BACKEND)
    list(APPEND HASHAPI_SOURCE_FILES src/hashapi/StubHashBackend.cpp)
else()
    list(APPEND HASHAPI_SOURCE_FILES
        src/Argon2idHasher.cpp
        src/hashapi/CpuHashBackend.cpp
    )
endif()

if(XENBLOCKS_BUILD_MINER)
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
    target_compile_definitions(${PROJECT_NAME} PRIVATE XENBLOCKS_BUILD_MINER=1)
    if(WIN32 AND XENBLOCKS_PAHO_MQTT_CPP_SHARED)
        target_compile_definitions(${PROJECT_NAME} PRIVATE PAHO_MQTTPP_IMPORTS)
    endif()
endif()

if(XENBLOCKS_BUILD_HASHAPI_CLI)
    add_executable(hashapi-cli
        src/hashapi/main.cpp
        ${HASHAPI_SOURCE_FILES}
    )
    target_include_directories(hashapi-cli PRIVATE src)
    if(XENBLOCKS_HASHAPI_STUB_BACKEND)
        target_compile_definitions(hashapi-cli PRIVATE XENBLOCKS_HASHAPI_STUB_BACKEND=1)
    else()
        if(unofficial-argon2_FOUND)
            target_link_libraries(hashapi-cli PRIVATE unofficial::argon2::libargon2)
        elseif(ARGON2_FOUND)
            target_include_directories(hashapi-cli PRIVATE ${ARGON2_INCLUDE_DIRS})
            target_link_libraries(hashapi-cli PRIVATE ${ARGON2_LIBRARIES})
        else()
            message(FATAL_ERROR "hashapi-cli requires libargon2 unless XENBLOCKS_HASHAPI_STUB_BACKEND=ON")
        endif()
    endif()
    if(nlohmann_json_FOUND)
        target_link_libraries(hashapi-cli PRIVATE nlohmann_json::nlohmann_json)
    endif()
endif()
