
//Code by Francis Jacob Kalliath 
//Project 2: Content-based Image Retrieval

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <dirent.h>
#include "csv_util.h"
#include <string.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include "features.h"
#include <algorithm>

using namespace cv;
using namespace std;


int main(int argc, char *argv[])
{
    std::vector<float> Tfeatures;
    std::vector<float> distance;
    char imagename[256];
    char filename[256];

    if (argc < 4)
    {
        printf("usage: %s image_name\n", argv[0]);
        exit(-1);
    }

    // get the directory path
    strcpy(imagename, argv[1]);
    Mat image = cv::imread(imagename);
    if (!strcmp(argv[2], "a")) 
    {  //function call for baseline
        Tfeatures = function_baseline(image);
    }
    else if (!strcmp(argv[2], "b")) 
    { //function call for color
        Tfeatures = function_histogram(image,8);
    }
    else if (!strcmp(argv[2], "c")) 
    { //function call for multi histograms of color
        Tfeatures = function_multiHist(image);
    }
    else if (!strcmp(argv[2], "d")) 
    { //function call for texture
        Tfeatures = function_texture(image);
    } 
    else if (!strcmp(argv[2], "e")) 
    { //function call for texture and color
        Tfeatures = function_textureColor(image);
    }
    else if (!strcmp(argv[2], "f")) 
    { //function call for color on middle part
        Mat mid = extractMiddle(image);
        Tfeatures = function_histogram(mid,8);
    }
    else if (!strcmp(argv[2], "g")) 
    { //function call for texture on middle part
        Mat mid = extractMiddle(image);
        Tfeatures = function_texture(mid);
    }
    else if (!strcmp(argv[2], "h")) 
    { //function call for texture and color on middle part
        Mat mid = extractMiddle(image);
        Tfeatures = function_textureColor(mid);
    }
    else if (!strcmp(argv[2], "i")) 
    { //function call for Gabor texture
        Tfeatures = function_gaborTexture(image);
    }
    else if (!strcmp(argv[2], "j"))  
    { //function call for Gabor texture and color
        Tfeatures = function_gaborTextureColor(image);
    }
    else if (!strcmp(argv[2], "k")) 
    { //function call for multi histograms of Gabor texture and color
        Tfeatures = function_multiGaborTextureColor(image);
    } 
    else if (!strcmp(argv[2], "l")) 
    { //function call for Gabor texture and color on middle part
        Mat mid = extractMiddle(image);
        Tfeatures = function_gaborTextureColor(mid);
    } 
    else
    {
        cout << "No such feature type." << endl;
        exit(-1);
    }

    // Compute the features of the image
    //ComputeFeatures(image, Tfeatures);
    strcpy(filename, "/home/francis-kalliath/PRCV_work/Project2_1/dataset/h.csv");
    std::vector<char *> filenames;
    std::vector<std::vector<float>> data;
    read_image_data_csv(filename, filenames, data, 0);
    float sum;
    
    pair<string, float> imageAndDistance;
    vector<pair<string, float>> distances;
    float d;
    for (int i = 0; i < filenames.size(); i++) {
        if (!strcmp(argv[3], "x")) {
            // Function call for sum of square difference
            d = SumofSquareDifference(Tfeatures, data[i]);
            imageAndDistance = make_pair(filenames[i], d);
            distances.push_back(imageAndDistance);
            // sort the vector of distances in ascending order
            sort(distances.begin(), distances.end(), [](auto &left, auto &right){
        return left.second < right.second;});
        } else if (!strcmp(argv[3], "y")) {
            // Function call for histogram intersection
            d = histogramIntersection(Tfeatures, data[i]);
            imageAndDistance = make_pair(filenames[i], d);
            distances.push_back(imageAndDistance);
            // sort the vector of distances in descending order
            sort(distances.begin(), distances.end(), [](auto &left, auto &right){
            return left.second < right.second;});
        } else {
            cout << "No such distance metrics." << endl;
            exit(-1);
        }
    }   
    for (int i = 0; i < distances.size() && (i < 11); i++) 
    {
        cout << distances[i].first<< std::endl;
    }
    printf("Terminating\n");
    return (0);
}
