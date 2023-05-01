//Project 4: Calibration and Augmented Reality
//Code written by Francis Jacob Kalliath

#ifndef PROJ3_PROCESSORS_H
#define PROJ3_PROCESSORS_H

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;
//creates a vector of 3d coordinates to use for 3d calibration
vector<Vec3f> coordinateConstructionTo3d(Size patternSize);

//the function identifies the corners of a chessboard in the given image
bool chessboardCornersIdentification(Mat &frame, Size patternSize, vector<Point2f> &corners);

//detects ArUco markers from a given frame and saves their coordinates in a vector of Point2f objects
bool arucoCornersIdentification(Mat &frame, vector<Point2f> &corners);

//draw circles on the corners of an image frame
void OutsideCorners(Mat &frame, vector<Vec3f> points, Mat rotationVector, Mat translationVector, Mat cameraMatrix, Mat distCoeffs);

// This method is used to draw the project on the target image
void VirtualObjectProjection(Mat &frame, Mat rotationVector, Mat translationVector, Mat cameraMatrix, Mat distCoeffs, int a);

//method to add the points to objectPoints
vector<Vec3f> ObjectPointStructure();
//method to add the points to objectPoints
vector<Vec3f> ObjectPointStructure1();
//method to add the points to objectPoints
vector<Vec3f> ObjectPointStructure2();
//method to add the points to objectPoints
vector<Vec3f> ObjectPointStructure3();
//method to add the points to objectPoints
vector<Vec3f> ObjectPointStructure4();
//method to add the points to objectPoints
vector<Vec3f> ObjectPointStructure5();

//method to draw line between the points
void drawLines(Mat &frame, vector<Point2f> p);

//this method overlays the image on the frame to create the displayedFrame
void overlayPicture(Mat &frame, Mat &displayedFrame, Mat &image);

//detect the markers on a frame and store their locations in a vector of Point2f objects
void arucoOutsidePoints(Mat &frame, vector<Point2f> &outsidePoints);

#endif
