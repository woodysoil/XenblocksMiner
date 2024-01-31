cmake_minimum_required(VERSION 3.27.9)
project(powsubmitter)
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
add_library(${PROJECT_NAME} SHARED ${SOURCES})
target_link_libraries(${PROJECT_NAME} ${PROJECT_LINK_LIBS} )
#Set the location for library installation -- i.e., /usr/lib in this case
# not really necessary in this example. Use "sudo make install" to apply
install(TARGETS ${PROJECT_NAME} DESTINATION /usr/lib)