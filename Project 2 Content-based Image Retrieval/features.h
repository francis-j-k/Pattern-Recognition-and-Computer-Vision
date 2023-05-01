
//Code by Francis Jacob Kalliath 
//Project 2: Content-based Image Retrieval

#ifndef PROJ2_FEATURES_H
#define PROJ2_FEATURES_H

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;



#endif //PROJ2_FEATURES_H
//function calculates the sum of square difference between the values of the two vectors
float SumofSquareDifference(std::vector<float> &target, std::vector<float> &database_vec );
// calculates the histogram intersection agrea to obtain distance between two features
float histogramIntersection(vector<float> &target, vector<float> &image);



//function calculates Color features for the 9x9 middle part of the image
vector<float> function_baseline(Mat &image);
////function calculates the image Color features 
int ComputeFeatures(Mat &image, vector<float> &features);
//function calculates the image Color histogram features 
vector<float> function_histogram(Mat &image,int s);
//function calculates the multi image Color histogram features 
vector<float> function_multiHist(Mat &image);
//function calculates the image Texture features 
vector<float> function_texture(Mat &image);
//function calculates the Texture and Color features 
vector<float> function_textureColor(Mat &image);
//function calculates the image gabor filter for Texture features 
vector<float> function_gaborTexture(Mat &image);
//function calculates the image gabor filter for Texture and Color features 
vector<float> function_gaborTextureColor(Mat &image);
//function calculates the multi image gabor filter for Texture and Color features 
vector<float> function_multiGaborTextureColor(Mat &image);




//function computes the sobel X filter
Mat CompSobelX(Mat &image);
//function computes the sobel X filter
Mat CompSobelY(Mat &image);
//function computes the gradient magnitude using sobel x and sobel y
Mat GradientMagnitude(Mat &image);
//function computes the gradient orientation
Mat Gradientorientation(Mat &image);
//function extracts the middle portion of the image 
Mat extractMiddle(Mat &image);
//function converts the mat values into vector
vector<float> matToVector(Mat &m);
