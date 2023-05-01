// Project 4: Calibration and Augmented RealityarucoLoc
// Code written by Francis Jacob Kalliath
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "processors.h"

using namespace cv;
using namespace std;

int main(int argc, char *argv[])
{
    Size chessboardSize(9, 6);
    Size arucoSize(5, 7);
    int NUMBER_CALIBRATION_FRAMES = 5;
    Mat chessboardDistCoeffs,arucoDistCoeffs;
    Mat chessboardMatrix, arucoMatrix;
    vector<Mat> chessboardvar1, chessboardvar2, arucovar1, arucovar2;
    vector<vector<Point2f>> chessboardListOfCorner;
    vector<vector<Vec3f>> chessboardListOfPoint;
    vector<vector<Point2f> > arucoListOfCorner;
    vector<vector<Vec3f> > arucoListOfPoints;
    vector<Vec3f> chessBoardLoc;
    vector<Vec3f> arucoLoc;
    Mat frame;
    vector<Point2f> cornersOfChessboard;
    vector<Point2f> cornersOfAruco;
    bool overLay = false;
    //Capturing video frame
    VideoCapture *capdev;
    capdev = new VideoCapture(0);
    if (!capdev->isOpened())
    {
        cout << "Error capturing the image ";
        return -1;
    }
    Mat image;
    image = imread(argv[1], 1);
    // validate image data
    if (!image.data) {
        cout << "No image data\n";
        return -1;
    }
    namedWindow("Video", 1);
    //Function call to constructing 3D Coordinates
    chessBoardLoc = coordinateConstructionTo3d(chessboardSize);
    arucoLoc = coordinateConstructionTo3d(arucoSize);

    while (true)
    {
        *capdev >> frame;
        if (frame.empty())
        {
            cout << "frame is empty\n";
            break;
        }
        resize(frame, frame, Size(), 0.6, 0.6);
        char key = waitKey(10);
        Mat displayedFrame = frame.clone();
        //detect the corners of a chessboard pattern
        
        bool iffoundChessboard = chessboardCornersIdentification(frame, chessboardSize, cornersOfChessboard);
        if (iffoundChessboard)
        {
            drawChessboardCorners(displayedFrame, chessboardSize, cornersOfChessboard, iffoundChessboard);
        }
        //detect the corners of a Aruco pattern
        bool iffoundAruco = arucoCornersIdentification(frame, cornersOfAruco);
        if (iffoundAruco) {
            for (int i = 0; i < cornersOfAruco.size(); i++) {
                circle(displayedFrame, cornersOfAruco[i], 1, Scalar(147, 200, 255), 2);
            }
        }
        bool overlayArucoCorners = arucoCornersIdentification(frame, cornersOfAruco);
        if (overlayArucoCorners) {
            // display the top left corner of each target
            for (int i = 0; i < cornersOfAruco.size(); i++) {

                circle(displayedFrame, cornersOfAruco[i], 1, Scalar(147, 200, 255), 2);
            }
        }
        //condition to overlay picture to Aruco target
        if (key == 'd') { 
            overLay = !overLay;
        }

        // apply the picture on ArUco target
        if (overLay && overlayArucoCorners) {
            overlayPicture(frame, displayedFrame, image);
        }

        // condition to calibrate the Chessboard
        if (key == 'a')
        {
            if (iffoundChessboard)
            {
                cout << "PLease select chessboard calibration image" << endl;
                chessboardListOfCorner.push_back(cornersOfChessboard);
                chessboardListOfPoint.push_back(chessBoardLoc);
            }
            else
            {
                cout << "Error finding the chessboard corners" << endl;
            }
        }
        // condition to calibrate the model with the chessboard images
        else if (key == 'z')
        {
            if (chessboardListOfPoint.size() < NUMBER_CALIBRATION_FRAMES)
            {
                cout << "Capture more that 5 image to calibrate. Not enough Images " << endl;
            }
            else
            {
                cout << "calibrate for camera done" << endl;
                double chessboardError = calibrateCamera(chessboardListOfPoint, chessboardListOfCorner, Size(frame.rows, frame.cols), chessboardMatrix, chessboardDistCoeffs, chessboardvar1, chessboardvar2);
                cout << "Matrix for Chessboard: " << endl;
                for (int i = 0; i < chessboardMatrix.rows; i++) {
                    for (int j = 0; j < chessboardMatrix.cols; j++) {
                        cout << chessboardMatrix.at<double>(i, j) << ", ";
                    }
                    cout << "\n";
                }
                cout << "Distortion Coefficients for the Chessboard: " << endl;
                for (int i = 0; i < chessboardDistCoeffs.rows; i++) {
                    for (int j = 0; j < chessboardDistCoeffs.cols; j++) {
                        cout << chessboardDistCoeffs.at<double>(i, j) << ", ";
                    }
                    cout << "\n";
                }
                cout << "Re-projection Error for Chessboard: " << chessboardError << endl;
            }
        }
        // condition to calibrate the Aruco 
        else if (key == 's') 
        {
            if (iffoundAruco) {
                cout << "Please select aruco calibration image" << endl;
                arucoListOfCorner.push_back(cornersOfAruco);
                arucoListOfPoints.push_back(arucoLoc);
            } else {
                cout << "No aruco corners found" << endl;
            }
        }
        // condition to calibrate the model with the Aruco images
        else if (key == 'x') 
        {
            if (arucoListOfPoints.size() < NUMBER_CALIBRATION_FRAMES) {
                cout << "Capture more that 5 image to calibrate. Not enough Images " << endl;
            } else {
                cout << "calibrate for camera done" << endl;
                double arucoError = calibrateCamera(arucoListOfPoints, arucoListOfCorner, Size(frame.rows, frame.cols), arucoMatrix, arucoDistCoeffs, arucovar1, arucovar2);
                cout << "Aruco Matrix: " << endl;
                for (int i = 0; i < arucoMatrix.rows; i++) {
                    for (int j = 0; j < arucoMatrix.cols; j++) {
                        cout << arucoMatrix.at<double>(i, j) << ", ";
                    }
                    cout << "\n";
                }
                cout << "Coefficients of Distortion for Aruco: " << endl;
                for (int i = 0; i < arucoDistCoeffs.rows; i++) {
                    for (int j = 0; j < arucoDistCoeffs.cols; j++) {
                        cout << arucoDistCoeffs.at<double>(i, j) << ", ";
                    }
                    cout << "\n";
                }
                cout << "Re-projection Error for Aruco: " << arucoError << endl;
            }
        }
        // condition to draw the projection on the chessboard 
        if (chessboardDistCoeffs.rows != 0)
        {
            vector<Point2f> currCorners;
            bool foundCurrCorners = chessboardCornersIdentification(frame, chessboardSize, currCorners);

            if (foundCurrCorners)
            {
                Mat rotationVector, translationVector;
                bool status = solvePnP(chessBoardLoc, currCorners, chessboardMatrix, chessboardDistCoeffs, rotationVector, translationVector);
                //cout<<"Rotation Vector"<<rotationVector<<endl;
                //cout<<"Translation Vector"<<translationVector<<endl;
                if (status)
                {
                    // drawing the AR object
                    OutsideCorners(displayedFrame, chessBoardLoc, rotationVector, translationVector, chessboardMatrix, chessboardDistCoeffs);
                    VirtualObjectProjection(displayedFrame, rotationVector, translationVector, chessboardMatrix, chessboardDistCoeffs,1);
                    VirtualObjectProjection(displayedFrame, rotationVector, translationVector, chessboardMatrix, chessboardDistCoeffs,2);
                    VirtualObjectProjection(displayedFrame, rotationVector, translationVector, chessboardMatrix, chessboardDistCoeffs,3);
                    VirtualObjectProjection(displayedFrame, rotationVector, translationVector, chessboardMatrix, chessboardDistCoeffs,4);
                    VirtualObjectProjection(displayedFrame, rotationVector, translationVector, chessboardMatrix, chessboardDistCoeffs,5);
                    VirtualObjectProjection(displayedFrame, rotationVector, translationVector, chessboardMatrix, chessboardDistCoeffs,6);
                }
            }
        }
        
        // condition to draw the projection on the Aruco
        if (arucoDistCoeffs.rows != 0) 
        {
            vector<Point2f> currCorners;
            bool foundCurrCorners = arucoCornersIdentification(frame, currCorners);

            if (foundCurrCorners) 
            {
                Mat rotationVector, translationVector;
                bool status = solvePnP(arucoLoc, currCorners, arucoMatrix, arucoDistCoeffs, rotationVector, translationVector);
                if (status) 
                {
                    // drawing the AR object
                    VirtualObjectProjection(displayedFrame, rotationVector, translationVector, arucoMatrix, arucoDistCoeffs,1);
                    VirtualObjectProjection(displayedFrame, rotationVector, translationVector, arucoMatrix, arucoDistCoeffs,2);
                    VirtualObjectProjection(displayedFrame, rotationVector, translationVector, arucoMatrix, arucoDistCoeffs,3);
                    VirtualObjectProjection(displayedFrame, rotationVector, translationVector, arucoMatrix, arucoDistCoeffs,4);
                    VirtualObjectProjection(displayedFrame, rotationVector, translationVector, arucoMatrix, arucoDistCoeffs,5);
                    //VirtualObjectProjection(displayedFrame, rotationVector, translationVector, arucoMatrix, arucoDistCoeffs,6);
                }
            }
        }


        namedWindow("Video", WINDOW_NORMAL);
        setWindowProperty("Video", WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);
        imshow("Video", displayedFrame);
        //hit "q" to end the camera
        if (key == 'q')
        {
            std::vector<std::vector<cv::Point_<float>>> chessboardCL = chessboardListOfCorner;
            break;
        }
    }
    return 0;
}