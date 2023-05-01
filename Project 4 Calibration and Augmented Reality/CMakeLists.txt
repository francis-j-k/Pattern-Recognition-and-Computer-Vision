#Project 4: Calibration and Augmented Reality
#Code written by Francis Jacob Kalliath

cmake_minimum_required(VERSION 3.16)

#set(CMAKE_CXX_STANDARD 14)
#set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

project(proj4)

set(OpenCV_DIR "/usr/local/Cellar/opencv/4.5.4_3/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)

set(PROJ3_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

include_directories(${PROJ3_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})

add_executable(proj4 AugmentedReality.cpp processors.cpp)
#add_executable(proj4 harrisCorner.cpp processors.cpp)

target_link_libraries(proj4 ${OpenCV_LIBS})