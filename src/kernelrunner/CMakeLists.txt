cmake_minimum_required(VERSION 3.27.9)
project(kernelrunner)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CUDA_ARCHITECTURES 75)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}) 
#Bring the headers, such as Student.h into the project
include_directories(../include)

#However, the file(GLOB...) allows for wildcard additions:
#file(GLOB SOURCES "src/Student.cpp")
file(GLOB SOURCES "./*.cpp")

#Generate the shared library from the sources
add_library(${PROJECT_NAME} SHARED ${SOURCES})

target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O2>
    $<$<COMPILE_LANGUAGE:CUDA>:-O2>
)

set_target_properties(${PROJECT_NAME} PROPERTIES 
    COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>"
)
#Set the location for library installation -- i.e., /usr/lib in this case
# not really necessary in this example. Use "sudo make install" to apply
install(TARGETS ${PROJECT_NAME} DESTINATION /usr/lib)

