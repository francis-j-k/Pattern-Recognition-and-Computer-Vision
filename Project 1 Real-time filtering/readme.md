# Foobar


Project 1 : Real time filtering 

## Operating system and IDE

Ubuntu 20.04.5 LTS
Visual Studio Code 1.74.3

## Instructions for running the Task 1(Reading an image and displaying it)

```c++

//in the CMakeLIst.txt copy the below code
cmake_minimum_required( VERSION 3.16)
project(check)

find_package(OpenCV 4)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 4 not found.")
   endif()

include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(check imgDisplay.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS} )


//In the Terminal type 
cd Project1
cd build

//to build 
cmake ..

// to Compile the program
make

// To run the program
./check
"q" Task 1 - quits the video streaming
"r" Extension - rotated by 90° clockwise direction
"a" Extension - rotated by 90° anti-clockwise direction
"o" Extension - image is resized

```


## Instructions for running your executables 2 to 9

```c++

//in the CMakeLIst.txt copy the below code
cmake_minimum_required( VERSION 3.16)
project(check)

find_package(OpenCV 4)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 4 not found.")
   endif()

include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(check vidDisplay.cpp filter.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS} )


cd Project1
cd build

//to build 
cmake ..

// to Compile the program
make

// To run the program
./check
"q" Task 2 - quits the video streaming
"g" Task 3 - grey scale
"h" Task 4 - custon grey scale
"b" Task 5 - blur
"x" Task 6 - sobel x
"y" Task 6 - sobel y
"m" Task 7 - gradient magnitude image
"l" Task 8 - blur quantization
"c" Task 9 - cartoonize
```

## Instructions for running the Task 10(modifying the Contrast and Brightness)

```c++
cd Project1
cd build

//to build 
cmake ..

// to Compile the program
make

// To run the program
./check
"w" Task 10 - Brigntness and Contrast of the Image
//in the terminal enter the alpha value
0.6
//hit enter
//enter the beta value
0.8
//hit enter


```

## Instructions for running the Extension Sparkling Negative colormap Image

```c++

cd Project1
cd build

//to build 
cmake ..

// to Compile the program
make

// To run the program
./check
"a" Extension Task - Sparkling Negative colormap Image
"q" Task 2 - quits the video streaming

```

## I have used 1 Time Travel Days out of 8