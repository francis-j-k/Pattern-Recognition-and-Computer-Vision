#Project 3: Real-time 2-D Object Recognition
#Code written by Francis Jacob Kalliath

cmake_minimum_required(VERSION 3.16)

#set(CMAKE_CXX_STANDARD 14)
#set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

project(objectRecognizer)

set(OpenCV_DIR "/usr/local/Cellar/opencv/4.5.4_3/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)

set(PROJ3_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include_directories(${PROJ3_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})

add_executable(objectRecognizer objectRecognizer.cpp processors.cpp)

target_link_libraries(objectRecognizer ${OpenCV_LIBS})