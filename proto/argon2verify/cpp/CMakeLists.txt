cmake_minimum_required(VERSION 3.18)
project(argon2Verify)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(PkgConfig REQUIRED)
pkg_check_modules(ARGON2 REQUIRED libargon2)

include_directories(${ARGON2_INCLUDE_DIRS})
link_directories(${ARGON2_LIBRARY_DIRS})

add_executable(argon2Verify main.cpp)
target_link_libraries(argon2Verify ${ARGON2_LIBRARIES})