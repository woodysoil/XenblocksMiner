cmake_minimum_required(VERSION 3.18)
project(argon2idHasher)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(PkgConfig REQUIRED)
pkg_check_modules(ARGON2 REQUIRED libargon2)

if(NOT ARGON2_FOUND)
    message(FATAL_ERROR "Libargon2 not found")
endif()

include_directories(${ARGON2_INCLUDE_DIRS})
link_directories(${ARGON2_LIBRARY_DIRS})

add_executable(${PROJECT_NAME} main.cpp Argon2idHasher.cpp)
target_link_libraries(${PROJECT_NAME} ${ARGON2_LIBRARIES})