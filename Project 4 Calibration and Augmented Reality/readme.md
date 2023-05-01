# Foobar

Code by Francis Jacob Kalliath 
Project 4: Calibration and Augmented Reality

## Operating system and IDE

Ubuntu 20.04.5 LTS
Visual Studio Code 1.74.3

Video for Calibration and projection of 3D Object:
https://drive.google.com/file/d/1QG-dgm_ieBDKlofQMkq4Mn1JBMLVMvCS/view?us
p=share_link
Video for Harris Corner detection:
https://drive.google.com/file/d/1KCk1bdZKkuAciwATj-c-jYgGON4UOcE1/view?us
p=share_link

## Instructions for running the Calibration and projection of the objects for all tasks

```c++

//in the CMakeLIst.txt copy the below code

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




//In the Terminal type 
cd Project4
cd build

//to build 
cmake ..

// to Compile the program
make

// To run the program
./proj4 <location of the image to be overlayed>
example:./proj4 /home/francis-kalliath/PRCV_work/Project4_all/Project4_1/Data/tiger.jpeg


"a" Task 2 - Select Calibration Images
"z" Task 3 - Calibrate the Camera

"s" Extension - Select Calibration Images for ArUco 
"x" Extension - Calibrate the Camera for ArUco 
"d" Extension - To overlay the selected image on ArUco
"q" to quit from the program execution




```


## Instructions for running Detect Robust Features
```c++

//in the CMakeLIst.txt copy the below code

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

#add_executable(proj4 AugmentedReality.cpp processors.cpp)
add_executable(proj4 harrisCorner.cpp processors.cpp)

target_link_libraries(proj4 ${OpenCV_LIBS})



//In the Terminal type 
cd Project4
cd build

//to build 
cmake ..

// to Compile the program
make

// To run the program
./proj4 
example: ./proj4

```


## I have used 1 Time Travel Days out of 5 remaining Time Travel Days
