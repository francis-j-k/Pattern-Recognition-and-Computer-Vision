#include <opencv2/opencv.hpp>

// method to convert rgp to greyscale
int convertToGray(const cv::Mat &frame, cv::Mat &filtered_frame);
// method to convert copy green color channel and replace it with the other two channel
int greyscale( cv::Mat &src, cv::Mat &dst );
// method to convert copy green color channel and replace it with the other two channel
int blur5x5(cv::Mat &src, cv::Mat &dst);
// method implimenting sobel x filter as separable 1x3 filter
int sobelX3x3( cv::Mat &src, cv::Mat &dst );
// method implimenting sobel y filter as separable 1x3 filter
int sobelY3x3( cv::Mat &src, cv::Mat &dst );
//method that generates gradient magnitude image from sobel x and sobel y
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst);
//method to impliment blur quantizes a color image
int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels);
//method to cartoonize a image
int cartoon( cv::Mat &src, cv::Mat&dst, int levels, int magThreshold );
// method to add sparkle and create the negative of the image
int addSparklesAndNegative( cv::Mat &src, cv::Mat &dst );
// Method to modify teh brightness and contrast
int BrightnessContrast(cv::Mat &src, cv::Mat &dst, double alpha, double beta);