# Foobar

Code by Francis Jacob Kalliath 
Project 2: Content-based Image Retrieval

## Operating system and IDE

Ubuntu 20.04.5 LTS
Visual Studio Code 1.74.3

## Instructions for running the Feature EXtraction for all tasks

```c++

//in the CMakeLIst.txt copy the below code

#Code by Francis Jacob Kalliath 
#Project 2: Content-based Image Retrieval

cmake_minimum_required( VERSION 3.16)
project(proj1)

find_package(OpenCV 4)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 4 not found.")
   endif()

include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(proj1 featureExtration.cpp features.cpp csv_util.cpp)
#add_executable(proj1 ImageRetrival.cpp features.cpp csv_util.cpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
#target_link_libraries(ret ${OpenCV_LIBS})




//In the Terminal type 
cd Project2
cd build

//to build 
cmake ..

// to Compile the program
make

// To run the program
./proj1 <location of the csv file to which the data should be written to> <location of dataset containing images> <enter the alphabet corresponding to type of feature extraction that has to be performed>
example:./proj1 /home/francis-kalliath/PRCV_work/Project2_1/dataset/a.csv /home/francis-kalliath/PRCV_work/Project2_1/dataset/olympus a

"a" Task 1 - baseline features model
"b" Task 2 - histogram features model
"c" Task 3 - multihistograms for color features model
"d" Task 4 - texture model
"e" Task 4 - texture and color model
"f" Task 4 - histogram features in the middle part of the image
"g" Task 4 - texture features in the middle part of the image
"h" Task 5 - texture and color features in the middle part of the image
"i" Extension - Gabor on texture features of the image 
"j" Extension - Gabor on texture and color features of the image 
"k" Extension - Gabor on multi-image texture and color features of the image 
"l" Extension - Gabor on texture and color features in the middle part of the image 



```


## Instructions for running Image Retrival for all the tasks
```c++

//in the CMakeLIst.txt copy the below code

#Code by Francis Jacob Kalliath 
#Project 2: Content-based Image Retrieval

cmake_minimum_required( VERSION 3.16)
project(proj1)

find_package(OpenCV 4)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 4 not found.")
   endif()

include_directories(${OpenCV_INCLUDE_DIRS})
#pragma endregionadd_executable(proj1 featureExtration.cpp features.cpp csv_util.cpp)
add_executable(proj1 ImageRetrival.cpp features.cpp csv_util.cpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
#target_link_libraries(ret ${OpenCV_LIBS})




//In the Terminal type 
cd Project2
cd build

//to build 
cmake ..

// to Compile the program
make

// To run the program
./proj1 <location of the target image> <enter the alphabet corresponding to type of feature extraction method that has to be performed> <enter the alphabet corresponding to the required distance metrics>
example: ./proj1 /home/francis-kalliath/PRCV_work/Project2_1/dataset/pic.0535.jpg a x

The following are the alphabet corresponding to type of feature extraction methods that has to be performed
"a" Task 1 - baseline features model
"b" Task 2 - histogram features model
"c" Task 3 - multihistograms for color features model
"d" Task 4 - texture model
"e" Task 4 - texture and color model
"f" Task 4 - histogram features in the middle part of the image
"g" Task 4 - texture features in the middle part of the image
"h" Task 5 - texture and color features in the middle part of the image
"i" Extension - Gabor on texture features of the image 
"j" Extension - Gabor on texture and color features of the image 
"k" Extension - Gabor on multi-image texture and color features of the image 
"l" Extension - Gabor on texture and color features in the middle part of the image 

The following are the alphabet corresponding to type of distance metrics methods that has to be performed
"x" - Sum of square difference
"y" - histogram intersection

```


## I have used 1 Time Travel Days out of 7 remaining Time Travel Days