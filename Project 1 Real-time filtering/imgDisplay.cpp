
#include <iostream>
#include <cstdio>  // lots of standard C/C++ functions including printf, scanf
#include <cstring> // C/C++ functions for working with strings (char arrays)
#include <opencv2/opencv.hpp>  // OpenCV main include file


int main(int argc, char* argv[])
{
    cv::Mat src;
    char filename[256];

    src = cv::imread("/home/francis-kalliath/PRCV/Test1/pic.jpg");
    if (src.data == NULL)
    { // check if imread was successful
        printf("Unable to read image %s\n", filename);
        exit(-2);
    }
    //createing a window to display the image
    cv::namedWindow("Display Image", cv::WindowFlags::WINDOW_NORMAL);
    //showing the image in the window
    cv::imshow("Display Image", src); 
    int key = 0;
    while (key != 'q') 
    {
        key = cv::waitKey(0);
        if (key == 'q') 
        {
            cv::destroyAllWindows();
            return 0;
        }
        //Rotating of the Image in 90 CLOCKWISE)
        else if (key == 'r') 
        {
            cv::Mat rotated;
            rotate(src, rotated, cv::RotateFlags::ROTATE_90_CLOCKWISE);
            cv::imshow("Display Image", rotated);
        }
        //Rotating of the Image in 90 COUNTERCLOCKWISE)
        else if (key == 'a')
        {
            cv::Mat rotated;
            rotate(src, rotated, cv::RotateFlags::ROTATE_90_COUNTERCLOCKWISE);
            cv::imshow("Display Image", rotated);
        }
        // Resizing of the image to orginal size
        else if (key == 'o') 
        {
            cv::Mat resized;
            resize(src, resized, cv::Size(), 1, 1);
            cv::imshow("Display Image", resized);
        }
    }
}