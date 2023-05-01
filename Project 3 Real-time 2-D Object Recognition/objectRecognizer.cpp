//Project 3: Real-time 2-D Object Recognition
//Code written by Francis Jacob Kalliath
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "processors.h"


using namespace cv;
using namespace std;


int main(int argc, char *argv[]) 
{
    if (argc < 3) {
        cout << "Incorrect number of inputs" << endl;
        exit(-1);
    }

    vector<string> ClassName;
    vector<vector<double>> features;
    Mat img;
    bool training = false;
    // loads data from a CSV file
    loadFromCSV(argv[1], ClassName, features);
    VideoCapture *capdev;
    capdev = new VideoCapture(0);
    //check if data is being fetched from webcam
    if (!capdev->isOpened()) 
    {
        cout << "Error:Unable to open webcam\n";
        return -1;
    }
    //Loop to capture and process every frame
    namedWindow("Webcam captured video", 1);
    while (true) 
    {
        *capdev >> img;
        if (img.empty()) {
            cout << "empty frame\n";
            break;
        }
        // q to quit the loop
        char val = waitKey(10);
        if (val == 'q') 
        {
            writeToCSV(argv[1], ClassName, features);
            break;
        }

        training = (val == 't') ? !training : training;
        cout << (training ? "Training" : "Testing") << endl;

        Mat thresholdFrame = thresholdAndCleanup(img);

        Mat Regions;
        Mat stats;
        Mat centroids;
        vector<int> CountTopLabels;
        Mat regionFrame = regions(thresholdFrame, Regions, stats, centroids, CountTopLabels);
        //loop to obtain the regions in the image
        for (int n = 0; n < CountTopLabels.size(); n++) 
        {
            int label = CountTopLabels[n];
            Mat region;
            region = (Regions == label);

            Moments m = moments(region, true);
            double CX = centroids.at<double>(label, 0);
            double CY = centroids.at<double>(label, 1);
            double theta = atan2(m.mu11, 0.5 * (m.mu20 - m.mu02));
            //function call to draw bounding box
            RotatedRect BoundingBox = obtainBoundingBox(region, CX, CY, theta);
            sketchLine(img, CX, CY, theta, Scalar(0, 0, 255));
            sketchBoundingBox(img, BoundingBox, Scalar(0, 255, 0));
            // function call to calculate the features
            vector<double> hu_Moments;
            calculateHu_Moments(m, hu_Moments);
            // condition for training the model
            if (training) 
            {
                namedWindow("Current Region", WINDOW_AUTOSIZE);
                imshow("Current Region", region);

                cout << "Enter the class that the image represents" << endl;
                char k = waitKey(0);
                string className = getClassid(k);

                features.push_back(hu_Moments);
                ClassName.push_back(className);

                if (n == CountTopLabels.size() - 1) 
                {
                    training = false;
                    cout << "Testing Mode" << endl;
                    destroyWindow("Current Region");
                }
            } else {
                // condition to do prediction of the model
                string className;
                if (!strcmp(argv[2], "n")) 
                {   // calling the classification methos
                    className = classification(features, ClassName, hu_Moments);
                } else if (!strcmp(argv[2], "k")) 
                {   // calling the KNearest Neighbour
                    className = classifierKNN(features, ClassName, hu_Moments, 5);
                }
                putText(img, className, Point(centroids.at<double>(label, 0), centroids.at<double>(label, 1)), FONT_HERSHEY_SIMPLEX, 2, Scalar(0, 0, 255), 3);
            }
        }

        imshow("Webcam captured video", img); 
    }
    return 0;
}