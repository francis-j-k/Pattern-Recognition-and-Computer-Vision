//main.cpp
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include "filter.h"
//#include "filter.cpp"

using namespace cv;
using namespace std;

int main(int argc, char** argv) { 
    VideoCapture cap(0);
    VideoWriter writer;
    double alpha, beta;
    if (!cap.isOpened()) {
        printf("Error: Unable to open video device\n");
        return  (-1);
    }

    // Creating a window to display the video
    namedWindow("Video", WINDOW_NORMAL);
    Mat frame, filteredFrame;

    bool isGray = false;
    bool isgreyscale = false;
    bool isGauss = false;
    bool isSobelx = false;
    bool isSobely = false;
    bool isGradMag = false;
    bool isblurQuantize = false;
    bool iscartoon = false;
    bool isaddSparklesAndNegative = false;
    bool isRecording = false;
    char key = 0;
    bool isBrightnessContrast = false;

    // While loop to continue until 'q' is pressed
    while (true) {
        // Capturing a new frame
        cap >> frame;
        // Checkking if the frame is empty
        if (frame.empty()) {
            printf("Error: Empty frame\n");
            break;
        }
        // Check for user input
        char key = (char)waitKey(10);
        if (key == 'q') {
            break;
        }
        //if loop to save the image
        else if (key == 's') {
            imwrite("saved_image.jpg", frame);
            cout << "Image saved as saved_image.jpg" << endl;
        }
        //if loop to check if the feed is recording
        else if (isRecording) {
            // writing the current frame to the video
            writer.write(frame);
        }
        //if loop to check if g is pressed
        else if (key == 'g') {
            isgreyscale = false;
            isGray = true;
            isGauss = false;
            isSobelx = false;
            isSobely = false;
            isGradMag = false;
            isblurQuantize = false;
            iscartoon = false;
            isaddSparklesAndNegative = false;
            isBrightnessContrast = false;
        }
        //if loop to check if h is pressed
        else if (key == 'h') {
            isgreyscale = true;
            isGray = false;
            isGauss = false;
            isSobelx = false;
            isSobely = false;
            isGradMag = false;
            isblurQuantize = false;
            iscartoon = false;
            isaddSparklesAndNegative = false;
            isBrightnessContrast = false;
        }
        //if loop to check if b is pressed
        else if (key == 'b') {
            isGray = false;
            isGauss = true;
            isgreyscale = false;
            isSobelx = false;
            isSobely = false;
            isGradMag = false;
            isblurQuantize = false;
            iscartoon = false;
            isaddSparklesAndNegative = false;
            isBrightnessContrast = false;
        }
        //if loop to check if x is pressed
        else if (key == 'x') {
            isgreyscale = false;
            isGray = false;
            isGauss = false;
            isSobelx = true;
            isSobely = false;
            isGradMag = false;
            isblurQuantize = false;
            iscartoon = false;
            isaddSparklesAndNegative = false;
            isBrightnessContrast = false;
        }//if loop to check if y is pressed
        else if (key == 'y') {
            isgreyscale = false;
            isGray = false;
            isGauss = false;
            isSobelx = false;
            isSobely = true;
            isGradMag = false;
            isblurQuantize = false;
            iscartoon = false;
            isaddSparklesAndNegative = false;
            isBrightnessContrast = false;
        }
        //if loop to check if m is pressed
        else if (key == 'm') {
            isgreyscale = false;
            isGray = false;
            isGauss = false;
            isSobelx = false;
            isSobely = false;
            isGradMag = true;
            isblurQuantize = false;
            iscartoon = false;
            isaddSparklesAndNegative = false;
            isBrightnessContrast = false;
        }
        //if loop to check if l is pressed
        else if (key == 'l') {
            isgreyscale = false;
            isGray = false;
            isGauss = false;
            isSobelx = false;
            isSobely = false;
            isGradMag = false;
            isblurQuantize = true;
            iscartoon = false;
            isaddSparklesAndNegative = false;
            isBrightnessContrast = false;
        }
        //if loop to check if c is pressed
        else if (key == 'c') {
            isgreyscale = false;
            isGray = false;
            isGauss = false;
            isSobelx = false;
            isSobely = false;
            isGradMag = false;
            isblurQuantize = false;
            iscartoon = true;
            isaddSparklesAndNegative = false;
            isBrightnessContrast = false;
        }
        //if loop to check if a is pressed
        else if (key == 'a') {
            isgreyscale = false;
            isGray = false;
            isGauss = false;
            isSobelx = false;
            isSobely = false;
            isGradMag = false;
            isblurQuantize = false;
            iscartoon = false;
            isaddSparklesAndNegative = true;
            isBrightnessContrast = false;
        }
        else if (key == 'w') {
            std::cout<<"Enter the value to adjust contrast: ";
            std::cin>>alpha;
            std::cout<<"Enter the value to adjust brightness: ";
            std::cin>>beta;
            isgreyscale = false;
            isGray = false;
            isGauss = false;
            isSobelx = false;
            isSobely = false;
            isGradMag = false;
            isblurQuantize = false;
            iscartoon = false;
            isaddSparklesAndNegative = false;
            isBrightnessContrast = true;
        }
        
        //if condition calling the grey scaling method
        if(isGray){
            convertToGray(frame, filteredFrame);
            // Display the greyscale frame
            imshow("Video", filteredFrame);
        }
        //if condition calling the alternative grey scaling method
        else if (isgreyscale) {
            greyscale(frame, filteredFrame);
            // Display the modified frame
            imshow("Video", filteredFrame);
        }
        //if condition calling the blurring method
        else if (isGauss) {
            blur5x5(frame, filteredFrame);
            // Display the modified frame
            cv::convertScaleAbs(filteredFrame,filteredFrame,2);
            imshow("Video", filteredFrame);
        }
        //if condition calling the sobelX method
        else if (isSobelx) {
            sobelX3x3(frame, filteredFrame);
            // Display the modified frame
            cv::convertScaleAbs(filteredFrame,filteredFrame,2);
            imshow("Video", filteredFrame);            
        }
        //if condition calling the sobelY method
        else if (isSobely) {
            sobelY3x3(frame, filteredFrame);
            // Display the modified frame
            cv::convertScaleAbs(filteredFrame,filteredFrame,2);
            imshow("Video", filteredFrame);            
        }
        //if condition calling the Gradient Magnitude method
        else if (isGradMag) {
            cv::Mat sx;
            sobelX3x3(frame, filteredFrame);
            filteredFrame.copyTo(sx);
            cv::Mat sy;
            sobelY3x3(frame, filteredFrame);
            filteredFrame.copyTo(sy);
            magnitude(sx, sy, filteredFrame);
            cv::convertScaleAbs(filteredFrame,filteredFrame,2);
            imshow("Video", filteredFrame);
        }
       //if condition calling the blur quantiz method 
        else if (isblurQuantize) {
            blurQuantize(frame, filteredFrame, 10);
            // Displaying the modified frame
            //cv::convertScaleAbs(filteredFrame,filteredFrame,2);
            imshow("Video", filteredFrame);            
        }
        //if condition calling the cartoon method
        else if (iscartoon) {

            cartoon(frame, filteredFrame, 10, 20);
            // Displaying the modified frame
            //cv::convertScaleAbs(filteredFrame,filteredFrame,2);
            imshow("Video", filteredFrame);            
        }
        //if condition calling the sSparkle and Negative method
        else if (isaddSparklesAndNegative) {
            addSparklesAndNegative(frame, filteredFrame);
            // Displaying the modified frame
            cv::convertScaleAbs(filteredFrame,filteredFrame,2);
            imshow("Video", filteredFrame);
        }
        else if (isBrightnessContrast) {
            BrightnessContrast(frame, filteredFrame, .5, 1);
            // Displaying the modified frame
            //cv::convertScaleAbs(filteredFrame,filteredFrame,2);
            imshow("Video", filteredFrame);
        }
        // check if the "o" key is pressed and will start recording
        else if (cv::waitKey(1) == 'o') {
            // start recording
            isRecording = true;
            // set up the video writer
            writer.open("output.avi", cv::VideoWriter::fourcc('M','J','P','G'), 30, frame.size(), true);
        }
        // check if the "p" key is pressed and will stop recording
        else if (cv::waitKey(1) == 'p') {
            // stop recording
            isRecording = false;
            writer.release();
        }
        else {
            // Display the color frame
            imshow("Video", frame);
        }
    }

    // Release the video channel and destroy the window
    cap.release();
    destroyAllWindows();

    return 0;
}