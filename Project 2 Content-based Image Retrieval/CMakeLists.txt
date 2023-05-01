
#Code by Francis Jacob Kalliath 
#Project 2: Content-based Image Retrieval

cmake_minimum_required( VERSION 3.16)
project(proj1)

find_package(OpenCV 4)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 4 not found.")
   endif()

include_directories(${OpenCV_INCLUDE_DIRS})
#add_executable(proj1 featureExtration.cpp features.cpp csv_util.cpp)
add_executable(proj1 ImageRetrival.cpp features.cpp csv_util.cpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
#target_link_libraries(ret ${OpenCV_LIBS})
