//Project 4: Calibration and Augmented Reality
//Code written by Francis Jacob Kalliath

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "processors.h"

using namespace std;
using namespace cv;

int main() {
    int size = 2;
    int k_size = 3;
    double a = 0.04;
    Mat gray;
    Mat frame;
    VideoCapture *capdev;
    capdev = new VideoCapture(0);
    if (!capdev->isOpened()) {
        cout << "Unable to open video device\n";
        return -1;
    }
    namedWindow("Video", 1);
    while (true) {
        *capdev >> frame;
        if (frame.empty()) {
            cout << "frame is empty\n";
            break;
        }

        char key = waitKey(10);
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        Mat dst = Mat::zeros(gray.size(), CV_32FC1);
        cornerHarris(gray, dst, size, k_size, a);

        double min_val, max_val;
        cv::minMaxLoc(dst, &min_val, &max_val);
        float thresh = 0.1 * max_val;
        for (int i = 0; i < dst.rows ; i++) {
            for(int j = 0; j < dst.cols; j++) {
                if (dst.at<float>(i,j) > thresh) {
                    circle(frame, Point(j,i), 1, Scalar(150, 19, 249), 2);
                }
            }
        }
        //press "q" to quit capturing frames for the web cam
        imshow("Video", frame);
        if (key == 'q') {
            break;
        }
    }
    destroyAllWindows();
    return 0;
}