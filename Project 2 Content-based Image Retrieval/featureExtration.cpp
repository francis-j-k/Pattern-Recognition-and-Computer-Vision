
//Code by Francis Jacob Kalliath 
//Project 2: Content-based Image Retrieval

#include <string.h>
#include <dirent.h>
#include <opencv2/opencv.hpp>
#include "csv_util.h"
#include "features.h"

using namespace cv;
using namespace std;

int main(int argc, char *argv[]) {
    char dirname[256];
    char filename[256];
    DIR *dirp;
    struct dirent *dp;
    //Input the location to with data should be written
    //strcpy(filename, "/home/francis-kalliath/PRCV_work/Project2_1/dataset/a.csv");
    strcpy(filename, argv[1]);

    // checkinf for sufficient arguments
    if (argc < 3) {
        cout << "Wrong input." << endl;
        exit(-1);
    }

    // getting the directory path
    strcpy(dirname, argv[2]);

    // opening the directory
    dirp = opendir(dirname);
    if (dirp == NULL) {
        cout << "Cannot open directory " << dirname << endl;
        exit(-1);
    }

    // loop over all the files in the directory
    while ((dp = readdir(dirp)) != NULL) {
        // check if the file is an image
        if(strstr(dp->d_name, ".jpg") || strstr(dp->d_name, ".png") ||
            strstr(dp->d_name, ".ppm") || strstr(dp->d_name, ".tif") ) {
            char buffer[256];
            strcpy(buffer, dirname);
            strcat(buffer, "/");
            strcat(buffer, dp->d_name);

            Mat image = imread(buffer);
            vector<float> imageFeature;

            if (!strcmp(argv[3], "a"))
            {// function call for baseline
                imageFeature = function_baseline(image);
                // ComputeFeatures(image, imageFeature);
            }
            else if (!strcmp(argv[3], "b"))
            {// function call for Histogram
                imageFeature = function_histogram(image,8);
            }
            else if (!strcmp(argv[3], "c")) 
            { // function call for multihistograms for color
                imageFeature = function_multiHist(image);
            }
            else if (!strcmp(argv[3], "d")) 
            { // function call for texture
                imageFeature = function_texture(image);
            } 
            else if (!strcmp(argv[3], "e")) 
            { // function call for texture and color
                imageFeature = function_textureColor(image);
            } 
            else if (!strcmp(argv[3], "f")) 
            { // color on middle part
                Mat mid = extractMiddle(image);
                imageFeature = function_histogram(mid,8);
            }
            else if (!strcmp(argv[3], "g")) 
            { // function call for texture on middle part
                Mat mid = extractMiddle(image);
                imageFeature = function_texture(mid);
            }
            else if (!strcmp(argv[3], "h")) 
            { // function call for texture and color on middle part
                Mat mid = extractMiddle(image);
                imageFeature = function_textureColor(mid);
            }
             else if (!strcmp(argv[3], "i")) 
            { // function call for Gabor texture
                imageFeature = function_gaborTexture(image);
            }
            else if (!strcmp(argv[3], "j"))  
            { // function call for Gabor texture and color
                imageFeature = function_gaborTextureColor(image);
            }
            else if (!strcmp(argv[3], "k")) 
            { // function call for multi histograms of Gabor texture and color
                imageFeature = function_multiGaborTextureColor(image);
            } 
            else if (!strcmp(argv[3], "l")) 
            { // function call for Gabor texture and color on middle part
                Mat mid = extractMiddle(image);
                imageFeature = function_gaborTextureColor(mid);
            } 
            else    
            {
                cout << "No such feature type." << endl;
                exit(-1);
            }
            append_image_data_csv(filename, buffer, imageFeature, 0);
        }
    }

}