cmake_minimum_required(VERSION 3.27.9)
project(appconfig)
set(CMAKE_BUILD_TYPE Release)

#Bring the headers, such as Student.h into the project
include_directories(../include)
#set ( PROJECT_LINK_LIBS libtestStudent.so )
file(GLOB PROJECT_LINK_LIBS "./*.so")
link_directories(/content/VMLearner/build)

#However, the file(GLOB...) allows for wildcard additions:
#file(GLOB SOURCES "src/Student.cpp")
file(GLOB SOURCES "./*.cpp")

#Generate the shared library from the sources
add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} ${PROJECT_LINK_LIBS} )
