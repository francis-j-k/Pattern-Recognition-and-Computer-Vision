# Foobar

Code by Francis Jacob Kalliath 
Project 3: Real-time 2-D Object Recognition

## Operating system and IDE

Ubuntu 20.04.5 LTS
Visual Studio Code 1.74.3

## Instructions for running the Feature EXtraction for all tasks

```c++

//in the CMakeLIst.txt copy the below code

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




//In the Terminal type 
cd Project3
cd build

//to build 
cmake ..

// to Compile the program
make

// To run the program
cd bin
./objectRecognizer <CSV file to which features should be writted into> <The type of classification that has to be performed('n' for classification and 'k' for KNN)>
example:./objectRecognizer featureDatabase.csv n

when you want to capture a image to train the model hit 't' then enter the char corresponding to the class having the name repeat the process to train the model

No extra step for extension

THe classemap are as follows:
{'a', "mug"}, {'b', "bottle"}, {'c', "bowl"}, {'d', "lotion"},
{'e', "scissor"}, {'f', "tiger"}, {'g', "train"}, {'h', "power bank"},
{'i', "specs"}, {'j', "clip"}

```

## I have used 1 Time Travel Days out of 6 remaining Time Travel Days