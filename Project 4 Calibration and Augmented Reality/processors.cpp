//Project 4: Calibration and Augmented Reality
//Code written by Francis Jacob Kalliath

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include "processors.h"

using namespace cv;
using namespace std;

//creates a vector of 3d coordinates to use for 3d calibration
vector<Vec3f> coordinateConstructionTo3d(Size boardSize) 
{
    vector<Vec3f> points(boardSize.height * boardSize.width);
    for (int i = 0; i < boardSize.height; i++) 
    {
        for (int j = 0; j < boardSize.width; j++) 
        {
            points[i * boardSize.width + j] = Vec3f(j, -i, 0);
        }
    }
    return points;
}

//the function identifies the corners of a chessboard in the given image
bool chessboardCornersIdentification(Mat &frame, Size boardSize, vector<Point2f> &corners) 
{
    // Identifies the corners
    bool foundCorners = findChessboardCorners(frame, boardSize, corners);
    if (foundCorners) 
    {
        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY); 
        Size PixWinSize(10, 10);
        TermCriteria termCrit(TermCriteria::COUNT|TermCriteria::EPS, 1, 0.1);
        //defines the corner positions by sub-pixel accuracy
        cornerSubPix(gray, corners, PixWinSize, Size(-1, -1), termCrit);
    }
    return foundCorners;
}

//detects ArUco markers from a given frame and saves their coordinates in a vector of Point2f objects
bool arucoCornersIdentification(Mat &frame, vector<Point2f> &corners) {
    corners.resize(35, Point2f(0, 0));
    vector<int> marker_Ids;
    vector<std::vector<cv::Point2f>> marker_Corners, rejectedPts;
    Ptr<cv::aruco::DetectorParameters> parameters = aruco::DetectorParameters::create();
    Ptr<cv::aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(aruco::DICT_6X6_250);
    //This function will detect the ArUco markers in the frame and populate the marker_Corners and marker_Ids vectors with the corresponding information.
    aruco::detectMarkers(frame, dictionary, marker_Corners, marker_Ids, parameters, rejectedPts);
    //loops through the marker_Ids vector and stores the coordinates of the first corner of each ArUco marker
    for (int i = 0; i < marker_Ids.size(); i++) {
        int idx = marker_Ids[i];
        corners[idx] = marker_Corners[i][0];
    }

    return marker_Corners.size() == 35;
}

//draw circles on the corners of an image frame
void OutsideCorners(Mat &frame, vector<Vec3f> points, Mat rotationVector, Mat translationVector, Mat cameraMatrix, Mat distCoeffs) {
    vector<Point2f> imagePoints;
    projectPoints(points, rotationVector, translationVector, cameraMatrix, distCoeffs, imagePoints);
    int index[] = {0, 8, 45, 53};
    for (int i : index) {
        circle(frame, imagePoints[i], 5, Scalar(150, 19, 249), 2);
    }
}

// This method is used to draw the project on the target image
void VirtualObjectProjection(Mat &frame, Mat rvec, Mat translationVector, Mat cameraMatrix, Mat distCoeffs, int a) {
    if (a==1)
    {   
        vector<Vec3f> objectPoints = ObjectPointStructure();
        vector<Point2f> projectedPoints;
        //projects 3D object points to the 2D image plane given intrinsic and extrinsic parameters of a camera
        projectPoints(objectPoints, rvec, translationVector, cameraMatrix, distCoeffs, projectedPoints);
        for (int i = 0; i < projectedPoints.size(); i++) {
            circle(frame, projectedPoints[i], 1, Scalar(150, 19, 249), 2);
        }
        //draws a line from the points stored in the projectedPoints
        drawLines(frame, projectedPoints);
    }
    else if (a==2)
    {   
        vector<Vec3f> objectPoints = ObjectPointStructure1();
        vector<Point2f> projectedPoints;
        //projects 3D object points to the 2D image plane given intrinsic and extrinsic parameters of a camera
        projectPoints(objectPoints, rvec, translationVector, cameraMatrix, distCoeffs, projectedPoints);
        for (int i = 0; i < projectedPoints.size(); i++) {
            circle(frame, projectedPoints[i], 1, Scalar(150, 19, 249), 2);
        }
        //draws a line from the points stored in the projectedPoints
        drawLines(frame, projectedPoints);
    }
    else if (a==3)
    {   
        vector<Vec3f> objectPoints = ObjectPointStructure2();
        vector<Point2f> projectedPoints;
        //projects 3D object points to the 2D image plane given intrinsic and extrinsic parameters of a camera
        projectPoints(objectPoints, rvec, translationVector, cameraMatrix, distCoeffs, projectedPoints);
        for (int i = 0; i < projectedPoints.size(); i++) {
            circle(frame, projectedPoints[i], 1, Scalar(150, 19, 249), 2);
        }
        //draws a line from the points stored in the projectedPoints
        drawLines(frame, projectedPoints);
    }
    else if (a==4)
    {   
        vector<Vec3f> objectPoints = ObjectPointStructure3();
        vector<Point2f> projectedPoints;
        //projects 3D object points to the 2D image plane given intrinsic and extrinsic parameters of a camera
        projectPoints(objectPoints, rvec, translationVector, cameraMatrix, distCoeffs, projectedPoints);
        for (int i = 0; i < projectedPoints.size(); i++) {
            circle(frame, projectedPoints[i], 1, Scalar(150, 19, 249), 2);
        }
        //draws a line from the points stored in the projectedPoints
        drawLines(frame, projectedPoints);
    }
    else if (a==5)
    {   
        vector<Vec3f> objectPoints = ObjectPointStructure4();
        vector<Point2f> projectedPoints;
        //projects 3D object points to the 2D image plane given intrinsic and extrinsic parameters of a camera
        projectPoints(objectPoints, rvec, translationVector, cameraMatrix, distCoeffs, projectedPoints);
        for (int i = 0; i < projectedPoints.size(); i++) {
            circle(frame, projectedPoints[i], 1, Scalar(150, 19, 249), 2);
        }
        //draws a line from the points stored in the projectedPoints
        drawLines(frame, projectedPoints);
    }
    else if (a==6)
    {   
        vector<Vec3f> objectPoints = ObjectPointStructure5();
        vector<Point2f> projectedPoints;
        //projects 3D object points to the 2D image plane given intrinsic and extrinsic parameters of a camera
        projectPoints(objectPoints, rvec, translationVector, cameraMatrix, distCoeffs, projectedPoints);
        for (int i = 0; i < projectedPoints.size(); i++) {
            circle(frame, projectedPoints[i], 1, Scalar(150, 19, 249), 2);
        }
        //draws a line from the points stored in the projectedPoints
        drawLines(frame, projectedPoints);
    }

}

//method to add the points to objectPoints
vector<Vec3f> ObjectPointStructure() {
    vector<Vec3f> objectPoints;
    objectPoints.push_back(Vec3f(4, -2, 1)); // Front bottom left
    objectPoints.push_back(Vec3f(5, -2, 1)); // Front bottom right
    objectPoints.push_back(Vec3f(4, -3, 1)); // Front top left
    objectPoints.push_back(Vec3f(5, -3, 1)); // Front top right
    objectPoints.push_back(Vec3f(4, -2, 4)); // Front bottom left
    objectPoints.push_back(Vec3f(5, -2, 4)); // Front bottom right
    objectPoints.push_back(Vec3f(4, -3, 4)); // Front top left
    objectPoints.push_back(Vec3f(5, -3, 4)); // Front top right
    return objectPoints;
}
//method to add the points to objectPoints
vector<Vec3f> ObjectPointStructure1() {
    vector<Vec3f> objectPoints;
    objectPoints.push_back(Vec3f(3, -2, 1)); // Front bottom left
    objectPoints.push_back(Vec3f(4, -2, 1)); // Front bottom right
    objectPoints.push_back(Vec3f(3, -3, 1)); // Front top left
    objectPoints.push_back(Vec3f(4, -3, 1)); // Front top right
    objectPoints.push_back(Vec3f(3, -2, 3)); // Front bottom left
    objectPoints.push_back(Vec3f(4, -2, 3)); // Front bottom right
    objectPoints.push_back(Vec3f(3, -3, 3)); // Front top left
    objectPoints.push_back(Vec3f(4, -3, 3)); // Front top right
    return objectPoints;
}
//method to add the points to objectPoints
vector<Vec3f> ObjectPointStructure2() {
    vector<Vec3f> objectPoints;
    objectPoints.push_back(Vec3f(2, -2, 1)); // Front bottom left
    objectPoints.push_back(Vec3f(3, -2, 1)); // Front bottom right
    objectPoints.push_back(Vec3f(2, -3, 1)); // Front top left
    objectPoints.push_back(Vec3f(3, -3, 1)); // Front top right
    objectPoints.push_back(Vec3f(2, -2, 2)); // Front bottom left
    objectPoints.push_back(Vec3f(3, -2, 2)); // Front bottom right
    objectPoints.push_back(Vec3f(2, -3, 2)); // Front top left
    objectPoints.push_back(Vec3f(3, -3, 2)); // Front top right
    return objectPoints;
}
//method to add the points to objectPoints
vector<Vec3f> ObjectPointStructure3() {
    vector<Vec3f> objectPoints;
    objectPoints.push_back(Vec3f(1, -2, 1)); // Front bottom left
    objectPoints.push_back(Vec3f(2, -2, 1)); // Front bottom right
    objectPoints.push_back(Vec3f(1, -3, 1)); // Front top left
    objectPoints.push_back(Vec3f(2, -3, 1)); // Front top right
    objectPoints.push_back(Vec3f(1, -2, 1)); // Front bottom left
    objectPoints.push_back(Vec3f(2, -2, 1)); // Front bottom right
    objectPoints.push_back(Vec3f(1, -3, 1)); // Front top left
    objectPoints.push_back(Vec3f(2, -3, 1)); // Front top right
    return objectPoints;
}
//method to add the points to objectPoints
vector<Vec3f> ObjectPointStructure4() {
    vector<Vec3f> objectPoints;
    objectPoints.push_back(Vec3f(5, -2, 1)); // Front bottom left
    objectPoints.push_back(Vec3f(6, -2, 1)); // Front bottom right
    objectPoints.push_back(Vec3f(5, -3, 1)); // Front top left
    objectPoints.push_back(Vec3f(6, -3, 1)); // Front top right
    objectPoints.push_back(Vec3f(5, -2, 5)); // Front bottom left
    objectPoints.push_back(Vec3f(6, -2, 5)); // Front bottom right
    objectPoints.push_back(Vec3f(5, -3, 5)); // Front top left
    objectPoints.push_back(Vec3f(6, -3, 5)); // Front top right
    return objectPoints;
}
//method to add the points to objectPoints
vector<Vec3f> ObjectPointStructure5() {
    vector<Vec3f> objectPoints;
    objectPoints.push_back(Vec3f(6, -2, 1)); // Front bottom left
    objectPoints.push_back(Vec3f(7, -2, 1)); // Front bottom right
    objectPoints.push_back(Vec3f(6, -3, 1)); // Front top left
    objectPoints.push_back(Vec3f(7, -3, 1)); // Front top right
    objectPoints.push_back(Vec3f(6, -2, 6)); // Front bottom left
    objectPoints.push_back(Vec3f(7, -2, 6)); // Front bottom right
    objectPoints.push_back(Vec3f(6, -3, 6)); // Front top left
    objectPoints.push_back(Vec3f(7, -3, 6)); // Front top right
    return objectPoints;
}
//method to draw line between the points
void drawLines(Mat &frame, vector<Point2f> p) {
    line(frame, p[0], p[1], Scalar(150, 19, 249), 2);
    line(frame, p[0], p[2], Scalar(150, 19, 249), 2);
    line(frame, p[1], p[3], Scalar(150, 19, 249), 2);
    line(frame, p[2], p[3], Scalar(150, 19, 249), 2);
    line(frame, p[4], p[6], Scalar(150, 19, 249), 2);
    line(frame, p[4], p[5], Scalar(150, 19, 249), 2);
    line(frame, p[5], p[7], Scalar(150, 19, 249), 2);
    line(frame, p[6], p[7], Scalar(150, 19, 249), 2);
    line(frame, p[0], p[4], Scalar(150, 19, 249), 2);
    line(frame, p[1], p[5], Scalar(150, 19, 249), 2);
    line(frame, p[2], p[6], Scalar(150, 19, 249), 2);
    line(frame, p[3], p[7], Scalar(150, 19, 249), 2);
}
//this method overlays the image on the frame to create the displayedFrame
void overlayPicture(Mat &frame, Mat &displayedFrame, Mat &image) {
    vector<Point2f> destination;
    arucoOutsidePoints(frame, destination);

    int height = image.size().height;
    int width = image.size().width;
    vector<Point2f> source;
    source.push_back(Point2f(0, 0));
    source.push_back(Point2f(width, 0));
    source.push_back(Point2f(0, height));
    source.push_back(Point2f(width, height));

    // Calculate Homography
    Mat Homography = findHomography(source, destination);
    // Warp source image to destination based on homography
    Mat output;
    warpPerspective(image, output, Homography, frame.size());
    for (int i = 0; i < output.rows; i++) {
        for (int j = 0; j < output.cols; j++) {
            if (output.at<Vec3b>(i, j) != Vec3b(0, 0, 0)) {
                displayedFrame.at<Vec3b>(i, j) = output.at<Vec3b>(i, j);
            }
        }
    }
}

//detect the markers on a frame and store their locations in a vector of Point2f objects
void arucoOutsidePoints(Mat &frame, vector<Point2f> &outsidePoints) {
    outsidePoints.resize(4, Point2f(0, 0));
    vector<int> marker_Ids;
    vector<std::vector<cv::Point2f>> marker_Corners, rejectedPts;
    Ptr<cv::aruco::DetectorParameters> parameters = aruco::DetectorParameters::create();
    Ptr<cv::aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(aruco::DICT_6X6_250);
    aruco::detectMarkers(frame, dictionary, marker_Corners, marker_Ids, parameters, rejectedPts);
    //All marker corner is then stored in the outsidePoints
    for (int i = 0; i < marker_Ids.size(); i++) {
        int idx = marker_Ids[i];
        if (idx == 30) {
            outsidePoints[0] = marker_Corners[i][3];
        } else if (idx == 0) {
            outsidePoints[1] = marker_Corners[i][0];
        } else if (idx == 34) {
            outsidePoints[2] = marker_Corners[i][2];
        } else if (idx == 4) {
            outsidePoints[3] = marker_Corners[i][1];
        }
    }
}