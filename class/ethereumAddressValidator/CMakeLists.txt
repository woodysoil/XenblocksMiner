cmake_minimum_required(VERSION 3.18)
project(ethereumAddressValidatorProject)

set(CMAKE_CXX_STANDARD 17)

find_library(CRYPTOPP_LIBRARY cryptopp)
if (NOT CRYPTOPP_LIBRARY)
    message(FATAL_ERROR "Crypto++ library not found")
endif()

add_executable(${PROJECT_NAME} main.cpp EthereumAddressValidator.cpp)
target_link_libraries(${PROJECT_NAME}  ${CRYPTOPP_LIBRARY})
