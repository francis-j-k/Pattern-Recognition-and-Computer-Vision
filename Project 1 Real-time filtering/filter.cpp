// filter.cpp

#include <opencv2/opencv.hpp>
#include "filter.h"

using namespace cv;
using namespace std;

// method to convert rgp to greyscale
int convertToGray(const cv::Mat &frame, cv::Mat &filtered_frame)
{
    cvtColor(frame, filtered_frame, COLOR_RGB2GRAY);
    return 0;
}

// method to convert copy green color channel and replace it with the other two channel
int greyscale(cv::Mat &src, cv::Mat &dst)
{
    if (src.empty())
        return -1;
    if (src.channels() != 3)
        return -1;
    dst = cv::Mat::zeros(src.size(), src.type());
    for (int i = 0; i < src.rows; i++)
    {
        for (int j = 0; j < src.cols; j++)
        {
            // access individual pixel
            cv::Vec3b &srcPixel = src.at<cv::Vec3b>(i, j);
            cv::Vec3b &dstPixel = dst.at<cv::Vec3b>(i, j);
            // copy green color channel to blue and red color channels
            dstPixel[0] = srcPixel[1];
            dstPixel[1] = srcPixel[1];
            dstPixel[2] = srcPixel[1];
        }
    }
    return 0; // success
}
// method implimenting the 5x5 gussian filter as separable 1x5 filters
int blur5x5(cv::Mat &src, cv::Mat &dst)
{
    dst = src.clone();

    // define the 1x5 filters
    int horizontalFilter[5] = {1, 2, 4, 2, 1};
    int verticalFilter[5] = {1, 2, 4, 2, 1};

    for (int c = 0; c < 3; c++)
    {
        for (int i = 2; i < src.rows - 2; i++)
        {
            for (int j = 2; j < src.cols - 2; j++)
            {
                int sum_h = 0;
                int sum_v = 0;

                // iterate through each element of the 1x5 filters
                for (int k = -2; k <= 2; k++)
                {
                    // apply the horizontal filter
                    sum_h += src.at<cv::Vec3b>(i, j + k)[c] * horizontalFilter[k + 2];

                    // apply the vertical filter
                    sum_v += src.at<cv::Vec3b>(i + k, j)[c] * verticalFilter[k + 2];
                }

                // set the blurred pixel value in the output image
                dst.at<cv::Vec3b>(i, j)[c] = (sum_h + sum_v) / 36;
            }
        }
    }

    return 0;
}

// method implimenting sobel x filter as separable 1x3 filter
int sobelX3x3(cv::Mat &src, cv::Mat &dst)
{
    dst = cv::Mat::zeros(src.rows, src.cols, CV_16SC3);

    // define the 1x3 filters
    int horizontalFilter[3] = {-1, 0, 1};
    int verticalFilter[3] = {1, 2, 1};

    cv::Mat temp1 = cv::Mat::zeros(src.rows, src.cols, CV_16SC3);
    cv::Mat temp2 = cv::Mat::zeros(src.rows, src.cols, CV_16SC3);

    // iterate through each channel of the image
    for (int c = 0; c < 3; c++)
    {
        // iterate through each pixel of the image
        for (int i = 1; i < src.rows - 1; i++)
        {
            for (int j = 1; j < src.cols - 1; j++)
            {
                int sum_h = 0;
                int sum_v = 0;

                // iterate through each element of the 1x3 filters
                for (int k = -1; k <= 1; k++)
                {
                    sum_h += src.at<cv::Vec3b>(i, j + k)[c] * horizontalFilter[k + 1];
                }
                temp1.at<cv::Vec3s>(i, j)[c] = sum_h;
            }
        }
    }

    // iterate through each channel of the image
    for (int c = 0; c < 3; c++)
    {
        // iterate through each pixel of the image
        for (int i = 1; i < temp1.rows - 1; i++)
        {
            for (int j = 1; j < temp1.cols - 1; j++)
            {
                int sum_v = 0;
                for (int k = -1; k <= 1; k++)
                {
                    sum_v += temp1.at<cv::Vec3s>(i + k, j)[c] * verticalFilter[k + 1];
                }
                temp2.at<cv::Vec3s>(i, j)[c] = sum_v;
            }
        }
    }
    temp2.convertTo(dst, -1, 0.25, 0);
    return 0;
}

// method implimenting sobel y filter as separable 1x3 filter
int sobelY3x3(cv::Mat &src, cv::Mat &dst)
{
    dst = cv::Mat::zeros(src.rows, src.cols, CV_16SC3);

    int horizontalFilter[3] = {1, 2, 1};
    int verticalFilter[3] = {-1, 0, 1};

    cv::Mat temp1 = cv::Mat::zeros(src.rows, src.cols, CV_16SC3);
    cv::Mat temp2 = cv::Mat::zeros(src.rows, src.cols, CV_16SC3);

    // iterate through each channel of the image
    for (int c = 0; c < 3; c++)
    {
        // iterate through each pixel of the image
        for (int i = 1; i < src.rows - 1; i++)
        {
            for (int j = 1; j < src.cols - 1; j++)
            {
                int sum_h = 0;
                int sum_v = 0;

                // iterate through each element of the 1x3 filters
                for (int k = -1; k <= 1; k++)
                {
                    sum_h += src.at<cv::Vec3b>(i, j + k)[c] * horizontalFilter[k + 1];
                }
                temp1.at<cv::Vec3s>(i, j)[c] = sum_h;
            }
        }
    }

    // iterate through each channel of the image
    for (int c = 0; c < 3; c++)
    {
        // iterate through each pixel of the image
        for (int i = 1; i < temp1.rows - 1; i++)
        {
            for (int j = 1; j < temp1.cols - 1; j++)
            {
                int sum_v = 0;

                // iterate through each element of the 3x1 filters
                for (int k = -1; k <= 1; k++)
                {
                    sum_v += temp1.at<cv::Vec3s>(i + k, j)[c] * verticalFilter[k + 1];
                }
                temp2.at<cv::Vec3s>(i, j)[c] = sum_v;
            }
        }
    }

    // Scale the final output
    temp2.convertTo(dst, -1, 0.25, 0);
    return 0;
}

//method that generates gradient magnitude image from sobel x and sobel y
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst)
{
    // Ensure the input images have the same size and type
    if (sx.size() != sy.size() || sx.type() != sy.type())
    {
        return -1;
    }
    cv::Mat magnitude(sx.size(), CV_32FC3);

    // Calculate the magnitude using Euclidean distance
    for (int y = 0; y < sx.rows; y++)
    {
        for (int x = 0; x < sx.cols; x++)
        {
            for (int c = 0; c < 3; c++)
            {
                float dx = sx.at<cv::Vec3s>(y, x)[c];
                float dy = sy.at<cv::Vec3s>(y, x)[c];
                magnitude.at<cv::Vec3f>(y, x)[c] = sqrt(dx * dx + dy * dy);
            }
        }
    }

    // Normalize the magnitude image to the range [0, 255]
    cv::normalize(magnitude, magnitude, 0, 255, cv::NORM_MINMAX, CV_8UC3);

    magnitude.convertTo(dst, CV_8UC3);
    return 0;
}

//method to impliment blur quantizes a color image
int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels)
{
    dst = cv::Mat::zeros(src.rows, src.cols, CV_8UC3);
    // blur image
    blur5x5(src, dst);
    // quantize image
    int b = 255 / levels;
    for (int i = 0; i < src.rows; i++)
    {
        for (int j = 0; j < src.cols; j++)
        {
            for (int k = 0; k < 3; k++)
            {
                int x = dst.at<cv::Vec3b>(i, j)[k];
                int xt = x / b;
                int xf = xt * b;
                dst.at<cv::Vec3b>(i, j)[k] = xf;
            }
        }
    }
    //src.copyTo(dst);
    // dst = src;
    return 0;
}

//method to cartoonize a image
int cartoon(cv::Mat &src, cv::Mat &dst, int levels, int magThreshold)
{
    // sobelx
    cv::Mat sx;
    sobelX3x3(src, sx);
    cv::Mat sy;
    // sobelx
    sobelY3x3(src, sy);
    cv::Mat mag_norm;
    // gradient magnitude
    magnitude(sx, sy, mag_norm);
    cv::convertScaleAbs(mag_norm, mag_norm, 2);
    cv::Mat blurquan;
    // blurr quantize
    blurQuantize(src, blurquan, levels);

    // allocate output image
    dst = cv::Mat::zeros(blurquan.rows, blurquan.cols, CV_8UC3);
    for (int i = 0; i <= blurquan.rows - 1; i++)
    {
        //Using pointers to refer to the pixels
        cv::Vec3b *dstrptr = dst.ptr<cv::Vec3b>(i);
        cv::Vec3b *mag_normrptr = mag_norm.ptr<cv::Vec3b>(i);
        cv::Vec3b *blurquanrptr = blurquan.ptr<cv::Vec3b>(i);
        //loop to access pixels
        for (int j = 0; j <= blurquan.cols - 1; j++)
        {
            for (int c = 0; c < 3; c++)
            {
                if (mag_normrptr[j][c] >= magThreshold)
                {
                    dstrptr[j][c] = 0;
                }
                else
                    dstrptr[j][c] = blurquanrptr[j][c];
            }
        }
    }
    return 0;
}

// method to add sparkle and create the negative of the image
int addSparklesAndNegative(cv::Mat &src, cv::Mat &dst)
{
    src.copyTo(dst);
    // Convert the image to grayscale
    Mat gray;
    cvtColor(dst, gray, cv::COLOR_BGR2GRAY);
    Mat edges;
    // edge detection using canny
    Canny(gray, edges, 50, 150);
    // Iterate over the image and add sparkles to areas with strong edges
    for (int y = 0; y < dst.rows; y++)
    {
        for (int x = 0; x < dst.cols; x++)
        {
            if (edges.at<uchar>(y, x) > 128)
            {
                dst.at<cv::Vec3b>(y, x)[0] = 255;
                dst.at<cv::Vec3b>(y, x)[1] = 255;
                dst.at<cv::Vec3b>(y, x)[2] = 255;
            }
        }
    }
    // Make the image negative
    for (int y = 0; y < dst.rows; y++)
    {
        for (int x = 0; x < dst.cols; x++)
        {
            dst.at<cv::Vec3b>(y, x)[0] = 255 - dst.at<cv::Vec3b>(y, x)[0];
            dst.at<cv::Vec3b>(y, x)[1] = 255 - dst.at<cv::Vec3b>(y, x)[1];
            dst.at<cv::Vec3b>(y, x)[2] = 255 - dst.at<cv::Vec3b>(y, x)[2];
        }
    }
    // Apply a color map to the image
    applyColorMap(dst, dst, cv::COLORMAP_JET);
    return 0;
}

// Method to modify teh brightness and contrast
int BrightnessContrast(cv::Mat &src, cv::Mat &dst, double alpha, double beta)
{
    // Modify the brightness and contrast of the frame
    dst = cv::Mat::zeros(src.rows, src.cols, CV_8UC3);
    for (int y = 0; y < src.rows; y++)
    {
        for (int x = 0; x < src.cols; x++)
        {
            for (int c = 0; c < src.channels(); c++)
            {
                // Update the pixel value
                if (alpha * src.at<cv::Vec3b>(y, x)[c] + beta >= 255)
                {
                    dst.at<cv::Vec3b>(y, x)[c] = 255;
                }
                else if (alpha * src.at<cv::Vec3b>(y, x)[c] + beta <= 0)
                {
                    dst.at<cv::Vec3b>(y, x)[c] = 0;
                }
                else
                {
                    // src.at<cv::Vec3b>(y, x)[c] = cv::saturate_cast<uchar>(alpha * src.at<cv::Vec3b>(y, x)[c] + beta);
                    dst.at<cv::Vec3b>(y, x)[c] = (alpha * src.at<cv::Vec3b>(y, x)[c] + beta);
                }
            }
        }
    }
    return 0;
}
