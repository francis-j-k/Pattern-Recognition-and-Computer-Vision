//Project 3: Real-time 2-D Object Recognition
//Code written by Francis Jacob Kalliath

#include <stdlib.h>
#include <map>
#include <float.h>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <string>
#include <vector>
#include "processors.h"

using namespace cv;
using namespace std;

//Method that caclulates the threshold and cleans up the image with inbuild functions 
Mat thresholdAndCleanup(Mat &image) {
    int THRESHOLD = 130;
    Mat processedImage;
    Mat grayscale;
    Mat processedImage1;
    processedImage = Mat(image.size(), CV_8UC1);

    cvtColor(image, grayscale, COLOR_BGR2GRAY);

    for (int i = 0; i < grayscale.rows; i++) {
        for (int j = 0; j < grayscale.cols; j++) {
            if (grayscale.at<uchar>(i, j) <= THRESHOLD) {
                processedImage.at<uchar>(i, j) = 255;
            } else {
                processedImage.at<uchar>(i, j) = 0;
            }
        }
    }

    const Mat kernel = getStructuringElement(MORPH_CROSS, Size(25, 25));
    morphologyEx(processedImage, processedImage1, MORPH_CLOSE, kernel);
    return processedImage1;
}

//Method that caclulates the threshold and cleans up the image 
Mat thresholdAndCleanup1(Mat &img) {
    Mat thresh_img(img.rows, img.cols, CV_8UC1);
    int threshold_value = 120;
    int max_value = 255;
    // char* key = argv[1];


    for (int i = 0; i < img.rows; i++)
    {
        for (int j = 0; j < img.cols; j++)
        {
            // Get the pixel value at (i,j)
            int pixel_value = img.at<uchar>(i, j);

            // Apply thresholding
            if (pixel_value < threshold_value)
            {
                thresh_img.at<uchar>(i, j) = max_value;
            }
            else
            {
                thresh_img.at<uchar>(i, j) = 0;
            }
        }
    }
    Mat dilatedImage = dilateImage(thresh_img, Size(3, 3), Point(-1, -1), 2);
    Mat erodedImage = erodeImage(dilatedImage, Size(3, 3), Point(-1, -1), 2);
    return erodedImage;
}

//Method that does dilation of the image
Mat dilateImage(Mat inputImage, Size kernelSize, Point anchor, int iterations)
{
    // Create structuring element for dilation
    Mat kernel = getStructuringElement(MORPH_CROSS, kernelSize, anchor);

    // Perform dilation
    Mat outputImage;
    dilate(inputImage, outputImage, kernel, anchor, iterations);

    return outputImage;
}

//Method that does erosion of image
Mat erodeImage(Mat inputImage, Size kernelSize, Point anchor, int iterations)
{
    // Create structuring element for erosion
    Mat kernel = getStructuringElement(MORPH_CROSS, kernelSize, anchor);

    // Perform erosion
    Mat outputImage;
    erode(inputImage, outputImage, kernel, anchor, iterations);

    return outputImage;
}

//Method that identifies the regions of the image
Mat regions(Mat &image, Mat &regionLabels, Mat &stats, Mat &centroids, vector<int> &topNLabels) {
    Mat processedImage;
    int componentLabels = connectedComponentsWithStats(image, regionLabels, stats, centroids);

    Mat areas = Mat::zeros(1, componentLabels - 1, CV_32S);
    Mat sortedIdx;
    for (int i = 1; i < componentLabels; i++) 
    {
        int area = stats.at<int>(i, CC_STAT_AREA);
        areas.at<int>(i - 1) = area;
    }
    if (areas.cols > 0) {
        sortIdx(areas, sortedIdx, SORT_EVERY_ROW + SORT_DESCENDING);
    }

    vector<Vec3b> colors(componentLabels, Vec3b(0, 0, 0));

    int N = 3;
    if (N < sortedIdx.cols) 
    {
        N = N;
    } else 
    {
        N = sortedIdx.cols;
    }
    int THRESHOLD = 4000;
    for (int i = 0; i < N; i++) 
    {
        int label = sortedIdx.at<int>(i) + 1;
        if (stats.at<int>(label, CC_STAT_AREA) > THRESHOLD) 
        {
            colors[label] = Vec3b(rand() % 256, rand() % 256, rand() % 256);
            topNLabels.push_back(label);
        }
    }

    processedImage = Mat::zeros(regionLabels.size(), CV_8UC3);
    for(int i = 0; i < processedImage.rows; i++) 
    {
        for (int j = 0; j < processedImage.cols; j++) 
        {
            int label = regionLabels.at<int>(i, j);
            processedImage.at<Vec3b>(i, j) = colors[label];
        }
    }
    return processedImage;
}

// Method that calculates the bounding box
RotatedRect obtainBoundingBox(Mat &region, double x, double y, double alpha) {
    int maxX = INT_MIN;
    int minX = INT_MAX;
    int maxY = INT_MIN;
    int minY = INT_MAX;
    for (int i = 0; i < region.rows; i++) {
        for (int j = 0; j < region.cols; j++) {
            if (region.at<uchar>(i, j) == 255) {
                int x1 = i * cos(alpha) - x * cos(alpha) + j * sin(alpha) - y * sin(alpha);
                int y1 = -i * sin(alpha) + x * sin(alpha) + j * cos(alpha) - y * cos(alpha);
                maxX = max(maxX, x1);
                minX = min(minX, x1);
                maxY = max(maxY, y1);
                minY = min(minY, y1);
            }
        }
    }
    int lx = maxX - minX;
    int ly = maxY - minY;
    if(lx<ly){
        int temp=lx;
        lx=ly;
        ly=temp;
    } 

    Point centroid = Point(x, y);
    Size size = Size(lx, ly);

    return RotatedRect(centroid, size, alpha * 180.0 / CV_PI);
}

// Method that draws the axis line
void sketchLine(Mat &image, double x, double y, double theta, Scalar color) {
    double l = 100.0;
    double xdot = x + sqrt(l * l - (l * sin(theta)) * (l * sin(theta)));
    double ydot = y + l * sin(theta);

    arrowedLine(image, Point(x, y), Point(xdot, ydot), color, 3);
}

// Method that draws the bounding box
void sketchBoundingBox(Mat &image, RotatedRect boundingBox, Scalar color) {
    Point2f rect_points[4];
    boundingBox.points(rect_points);
    for (int i = 0; i < 4; i++) {
        Point start = rect_points[i];
        Point end = rect_points[(i + 1) % 4];
        // Draw the line on the image
        line(image, start, end, color, 3);
    }
}

// Method that calculates the features
void calculateHu_Moments(Moments mo, vector<double> &hu_Moments) {
    double huMo[7]; 
    //double d;
    HuMoments(mo, huMo);
    for ( double d : huMo) {
        hu_Moments.push_back(d);
    }
    return;
}

// Method that calculates the Euclidean Distance
double computeEuclideanDistance(vector<double> feats1, vector<double> feats2) {
    double sumfeats1 = 0;
    double sumfeats2 = 0;
    double sumfeatsDifference = 0;
    for (int i = 0; i < feats1.size(); i++) {
        double diff = feats1[i] - feats2[i];
        // Square the difference and add it to the sum of squared differences
        double squaredDiff = diff * diff;
        sumfeatsDifference += squaredDiff;
        sumfeats1 = sumfeats1 + feats1[i] * feats1[i];
        sumfeats2 = sumfeats2 + feats2[i] * feats2[i];
    }
    double eucliduan = sqrt(sumfeatsDifference) / (sqrt(sumfeats1) + sqrt(sumfeats2));
    return eucliduan;
}


// Method that compares the features of the frame withe stored data
string classification(vector<vector<double>> features, vector<string> ClassNames, vector<double> hu_Moments) {
    double THRESHOLD = 0.2;
    double dist = DBL_MAX;
    string name = " ";
    for (int i = 0; i < features.size(); i++) {
        vector<double> dbFeature = features[i];
        string storedClassName = ClassNames[i];
        double eucDistance = computeEuclideanDistance(dbFeature, hu_Moments);
        if (eucDistance < dist && eucDistance < THRESHOLD) {
            name = storedClassName;
            dist = eucDistance;
        }
    }
    return name;
}

// Method that compares the features of the frame withe stored data using K Nearest Neighbour
string classifierKNN(vector<vector<double>> features, vector<string> ClassNames, vector<double> hu_Moments, int K) {
    double THRESHOLD = 0.15;
    vector<double> dist;
    for (int i = 0; i < features.size(); i++) {
        vector<double> eachFeature = features[i];
        double euc_dist = computeEuclideanDistance(eachFeature, hu_Moments);
        if (euc_dist < THRESHOLD) {
            dist.push_back(euc_dist);
        }
    }

    string className = " ";
    if (dist.size() > 0) {
        int n = dist.size();
        vector<int> sortedIdx(n);
        for (int i = 0; i < n; i++) {
            sortedIdx[i] = i;
        }

        for (int i = 0; i < n-1; i++) {
            for (int j = 0; j < n-i-1; j++) {
                if (dist[sortedIdx[j]] > dist[sortedIdx[j+1]]) {
                    swap(sortedIdx[j], sortedIdx[j+1]);
                }
            }
        }

        vector<string> firstKNames;
        int s = sortedIdx.size();
        vector<int> nameCount(ClassNames.size(), 0);
        int range = min(s, K);
        for (int i = 0; i < range; i++) {
            string name = ClassNames[sortedIdx[i]];
            int nameIdx = find(ClassNames.begin(), ClassNames.end(), name) - ClassNames.begin();
            nameCount[nameIdx]++;
        }

        int maxCount = 0;
        for (int i = 0; i < nameCount.size(); i++) {
            if (nameCount[i] > maxCount) {
                className = ClassNames[i];
                maxCount = nameCount[i];
            }
        }
    }
    return className;
}

//Method that takes the class name and id
string getClassid(char a) {
    std::map<char, string> myMap {
            {'a', "mug"}, {'b', "bottle"}, {'c', "bowl"}, {'d', "lotion"},
            {'e', "scissor"}, {'f', "tiger"}, {'g', "train"}, {'h', "power bank"},
            {'i', "specs"}, {'j', "clip"}
    };
    return myMap[a];
}


// Method that writes the features into the CSV file
void writeToCSV(string filename, vector<string> classNamesDB, vector<vector<double>> featuresDB) {
    
    ofstream csvFile;
    csvFile.open(filename, ofstream::trunc);

    for (int i = 0; i < classNamesDB.size(); i++) {
        csvFile << classNamesDB[i] << ",";
        for (int j = 0; j < featuresDB[i].size(); j++) {
            csvFile << featuresDB[i][j];
            if (j != featuresDB[i].size() - 1) {
                csvFile << ",";
            }
        }
        csvFile << "\n";
    }
}

// Method that loads the data from the CSV file to do prediction
void loadFromCSV(string filename, vector<string> &classNamesDB, vector<vector<double>> &featuresDB) {

    ifstream csvFile(filename);
    if (csvFile.is_open()) {

        string line;
        while (getline(csvFile, line)) {
            vector<string> currLine; 
            int pos = 0;
            string token;
            while ((pos = line.find(",")) != string::npos) {
                token = line.substr(0, pos);
                currLine.push_back(token);
                line.erase(0, pos + 1);
            }
            currLine.push_back(line);

            vector<double> currFeature;
            if (currLine.size() != 0) {
                classNamesDB.push_back(currLine[0]);
                for (int i = 1; i < currLine.size(); i++) {
                    currFeature.push_back(stod(currLine[i]));
                }
                featuresDB.push_back(currFeature);
            }
        }
    }
}
