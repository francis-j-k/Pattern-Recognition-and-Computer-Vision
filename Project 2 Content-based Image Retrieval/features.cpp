
//Code by Francis Jacob Kalliath 
//Project 2: Content-based Image Retrieval

#include <math.h>
#include <opencv2/opencv.hpp>
#include "features.h"

using namespace cv;
using namespace std;


//function to take an image use 9x9 squares of the middle part and create a feature vector
int ComputeFeatures(Mat &image, vector<float> &features) 
{

  if (image.empty()) {
    cout << "Error: The input image is empty." << endl;
    return (0);
  }

  int rows = image.rows;
  int cols = image.cols;
  // Assigning the sart and end positions of the middle portion of the image
  int startRow = (rows/2)-4;
  int endRow = startRow + 9;
  int startCol = (cols/2)-4;
  int endCol = startCol + 9;

  for (int i = startRow; i < endRow; i++) {
    for (int j = startCol; j < endCol; j++) {
      for (int c = 0; c < 3; c++) {
        cout<<image.at<Vec3b>(i, j)[c]<<" ";
        features.push_back(image.at<Vec3b>(i, j)[c]);
      }
    }
  }
  return (0);
}

//function to take an image use 9x9 squares of the middle part and extract feature
vector<float> function_baseline(Mat &image) 
{
    Mat middle9X9;
    int x = image.cols / 2 - 4;
    int y = image.rows / 2 - 4;
    middle9X9 = image(Rect(x, y, 9, 9)).clone();
    return matToVector(middle9X9);
}


//function converts Mat to vector float
vector<float> matToVector(Mat &m) 
{
    Mat flat = m.reshape(1, m.total() * m.channels());
    flat.convertTo(flat, CV_32F);
    return m.isContinuous() ? flat : flat.clone();
}


float SumofSquareDifference(std::vector<float> &target, std::vector<float> &database_vec ){

    float sum = 0;
    //looping to calculate the sum of square difference
    for (int i = 0; i < target.size(); i ++) {
        sum = sum +sqrt ((target[i] - database_vec[i]) * (target[i] - database_vec[i]));
    }
    //Normalize
    sum=sum/database_vec.size();
    return sum;
}

vector<float> function_histogram(Mat &image, int s) {

    // initialize a 3D histogram
    int HSize[] = {s, s, s};
    Mat f = Mat::zeros(3, HSize, CV_32F);

    // loop the image and build a 3D histogram
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            int b = image.at<Vec3b>(i, j)[0] / (256 / 8);
            int g = image.at<Vec3b>(i, j)[1] / (256 / 8);
            int r = image.at<Vec3b>(i, j)[2] / (256 / 8);
            f.at<float>(b, g, r)++;
        }
    }
    normalize(f, f, 1, 0, NORM_L2, -1, Mat());
    return (matToVector(f));
}

// function to obtain the histogram intersection
float histogramIntersection(vector<float> &target, vector<float> &image) {
    if (target.size() != image.size()) {
        throw runtime_error("Error: target and image vectors have different sizes.");
    }
    //chech if the sizes are same
    CV_Assert(target.size() == image.size());
    float inte = 0;
    //to obtain the min val
    for (int i = 0; i < target.size(); i++) {
        inte += (min(target[i], image[i]));
    }
    return (1.0-inte);
}

//Function to obtain the multi histogram color feature 
vector<float> function_multiHist(Mat &image) {
    vector<float> fea;
    int X1[] = {0, image.cols / 2};
    int Y1[] = {0, image.rows / 2};
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            Mat m = image(Rect(X1[i], Y1[j], image.cols / 2, image.rows / 2)).clone();
            vector<float> v = function_histogram(m,8);
            fea.insert(fea.end(), v.begin(), v.end()); 
        }
    }
    return fea;
}

//Function to obtain the texture feature in a vector
vector<float> function_texture(Mat &image)
{

    Mat grayscale;
    cvtColor(image, grayscale, COLOR_BGR2GRAY);
    Mat imaMag = GradientMagnitude(grayscale);
    Mat imaOri = Gradientorientation(grayscale);

    int histSize[] = {8, 8};
    Mat fea = Mat::zeros(2, histSize, CV_32F);

    float rangeMag = 400 / 8.0;
    float rangeOri = 2 * CV_PI / 8.0;

    for (int i = 0; i < imaMag.rows; i++)
    {
        for (int j = 0; j < imaMag.cols; j++)
        {
            int m = imaMag.at<float>(i, j) / rangeMag;
            int o = (imaOri.at<float>(i, j) + CV_PI) / rangeOri;
            fea.at<float>(m, o)++;
        }
    }
    normalize(fea, fea, 1, 0, NORM_L2, -1, Mat());
    return matToVector(fea);
}

//Function to obtain the texture and color feature in a vector
vector<float> function_textureColor(Mat &image)
{
    vector<float> fea = function_texture(image);
    vector<float> color = function_histogram(image,8);
    fea.insert(fea.end(), color.begin(), color.end());
    return fea;
}

//Function to obtain the Image Gabor texture feature in a vector
vector<float> function_gaborTexture(Mat &image) {
    vector<float> fea;

    // convert image to grayscale
    Mat grayscale;
    cvtColor(image, grayscale, COLOR_BGR2GRAY);

    float sigmaValue[] = {1.0, 2.0, 4.0};
    for (auto s : sigmaValue) {
        for (int k = 0; k < 16; k++) {
            float t = k * CV_PI / 8;
            Mat gaborKernel = getGaborKernel( Size(31,31), s, t, 10.0, 0.5, 0, CV_32F );
            Mat filImg;
            vector<float> hist(9, 0);
            filter2D(grayscale, filImg, CV_32F, gaborKernel);

            Scalar mean, stddev;
            meanStdDev(filImg, mean, stddev);
            fea.push_back(mean[0]);
            fea.push_back(stddev[0]);
        }
    }
    //normalizing
    normalize(fea, fea, 1, 0, NORM_L2, -1, Mat());

    return fea;
}

//Function to obtain the Image Gabor texture and color feature in a vector
vector<float> function_gaborTextureColor(Mat &image) {
    vector<float> fea = function_gaborTexture(image);
    vector<float> color = function_histogram(image,8);
    fea.insert(fea.end(), color.begin(), color.end());
    return fea;

}

//Function to obtain the multi Image Gabor texture and color feature in a vector
vector<float> function_multiGaborTextureColor(Mat &image) {
    vector<float> fea;
    int X[] = {0, image.cols / 2};
    int Y[] = {0, image.rows / 2};
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            Mat m = image(Rect(X[i], Y[j], image.cols / 2, image.rows / 2)).clone();
            vector<float> a = function_gaborTextureColor(m); 
            fea.insert(fea.end(), a.begin(), a.end()); 
        }
    }
    return fea;
}


// function to obtain the sobel X
Mat CompSobelX(Mat &image)
{
    Mat fea = Mat::zeros(image.size(), CV_32F);
    Mat temp = Mat::zeros(image.size(), CV_32F);

    // apply horizontal filter
    for (int i = 0; i < image.rows; i++)
    {
        for (int j = 0; j < image.cols; j++)
        {
            if (j > 0 && j < image.cols - 1)
            {
                temp.at<float>(i, j) = -image.at<uchar>(i, j - 1) + image.at<uchar>(i, j + 1);
            }
        }
    }
    // apply vertical filter
    for (int i = 0; i < temp.rows; i++)
    {
        for (int j = 0; j < temp.cols; j++)
        {
            if (i == 0)
            {
                fea.at<float>(i, j) = (2 * temp.at<float>(i, j) + temp.at<float>(i + 1, j) + temp.at<float>(i + 1, j)) / 4;
            }
            else if (i == temp.rows - 1)
            {
                fea.at<float>(i, j) = (2 * temp.at<float>(i, j) + temp.at<float>(i - 1, j) + temp.at<float>(i - 1, j)) / 4;
            }
            else
            {
                fea.at<float>(i, j) = (2 * temp.at<float>(i, j) + temp.at<float>(i + 1, j) + temp.at<float>(i - 1, j)) / 4;
            }
        }
    }
    return fea;
}

// function to obtain the sobel Y
Mat CompSobelY(Mat &image)
{
    Mat fea = Mat::zeros(image.size(), CV_32F);
    Mat temp = Mat::zeros(image.size(), CV_32F);

    // apply horizontal filter
    for (int i = 0; i < image.rows; i++)
    {
        for (int j = 0; j < image.cols; j++)
        {
            if (j == 0)
            {
                temp.at<float>(i, j) = (2 * image.at<uchar>(i, j) + image.at<uchar>(i, j + 1) + image.at<uchar>(i, j + 1)) / 4;
            }
            else if (j == image.cols - 1)
            {
                temp.at<float>(i, j) = (2 * image.at<uchar>(i, j) + image.at<uchar>(i, j - 1) + image.at<uchar>(i, j - 1)) / 4;
            }
            else
            {
                temp.at<float>(i, j) = (2 * image.at<uchar>(i, j) + image.at<uchar>(i, j + 1) + image.at<uchar>(i, j - 1)) / 4;
            }
        }
    }
    // apply vertical filter
    for (int i = 0; i < temp.rows; i++)
    {
        for (int j = 0; j < temp.cols; j++)
        {
            if (i > 0 && i < temp.rows - 1)
            {
                fea.at<float>(i, j) = -temp.at<float>(i - 1, j) + temp.at<float>(i + 1, j);
            }
        }
    }
    return fea;
}

// function to obtain GRadient Mangnitude
Mat GradientMagnitude(Mat &image)
{
    
    Mat x = CompSobelX(image);
    Mat y = CompSobelY(image);

    Mat fea;
    sqrt(x.mul(x) + y.mul(y), fea);

    return fea;
}

// function to obtain the gradient orientation
Mat Gradientorientation(Mat &image)
{
    
    Mat x = CompSobelX(image);
    Mat y = CompSobelY(image);

    Mat fea(image.size(), CV_32F);
    for (int i = 0; i < image.rows; i++)
    {
        for (int j = 0; j < image.cols; j++)
        {
            fea.at<float>(i, j) = atan2(y.at<float>(i, j), x.at<float>(i, j));
        }
    }

    return fea;
}


//function to extract the middle portion of the image
Mat extractMiddle(Mat &image) {
    Mat middle = image(Rect(image.cols / 3, image.rows / 3, image.cols / 3, image.rows / 3)).clone();
    return middle;
}