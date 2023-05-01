//Project 3: Real-time 2-D Object Recognition
//Code written by Francis Jacob Kalliath

#ifndef PROJ3_PROCESSORS_H
#define PROJ3_PROCESSORS_H

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

//Method that caclulates the threshold and cleans up the image with inbuild functions 
Mat thresholdAndCleanup(Mat &image);

//Method that caclulates the threshold and cleans up the image 
Mat thresholdAndCleanup1(Mat &image);

//Method that does dilation of the image
Mat dilateImage(Mat inputImage, Size kernelSize, Point anchor, int iterations);

//Method that does erosion of image
Mat erodeImage(Mat inputImage, Size kernelSize, Point anchor, int iterations);

//Method that identifies the regions of the image
Mat regions(Mat &image, Mat &labeledRegions, Mat &stats, Mat &centroids, vector<int> &topNLabels);

// Method that calculates the bounding box
RotatedRect obtainBoundingBox(Mat &region, double x, double y, double theta);

// Method that draws the axis line
void sketchLine(Mat &image, double x, double y, double theta, Scalar color);

// Method that draws the bounding box
void sketchBoundingBox(Mat &image, RotatedRect boundingBox, Scalar color);

// Method that calculates the features
void calculateHu_Moments(Moments mo, vector<double> &hu_Moments);

// Method that calculates the Euclidean Distance
double computeEuclideanDistance(vector<double> feats1, vector<double> feats2);

// Method that compares the features of the frame withe stored data
string classification(vector<vector<double>> featureVectors, vector<string> classNames, vector<double> currentFeature);

// Method that compares the features of the frame withe stored data using K Nearest Neighbour
string classifierKNN(vector<vector<double>> featureVectors, vector<string> classNames, vector<double> currentFeature, int K);

//Method that takes the class name and id
string getClassid(char a);

// Method that writes the features into the CSV file
void writeToCSV(string filename, vector<string> classNamesDB, vector<vector<double>> featuresDB);

// Method that loads the data from the CSV file to do prediction
void loadFromCSV(string filename, vector<string> &classNamesDB, vector<vector<double>> &featuresDB);

#endif //PROJ3_PROCESSORS_H
