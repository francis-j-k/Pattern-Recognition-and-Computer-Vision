cmake_minimum_required( VERSION 3.16)
project(check)

find_package(OpenCV 4)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 4 not found.")
   endif()

include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(check vidDisplay.cpp filter.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS} )