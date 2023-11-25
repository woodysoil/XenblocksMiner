cmake_minimum_required(VERSION 3.18)
project(EthereumAddressValidator)

set(CMAKE_CXX_STANDARD 14)

find_library(CRYPTOPP_LIBRARY cryptopp)
if (CRYPTOPP_LIBRARY)
    add_executable(ethereumAddressValidator main.cpp)
    target_link_libraries(ethereumAddressValidator ${CRYPTOPP_LIBRARY})
else()
    message(FATAL_ERROR "Crypto++ library not found")
endif()

