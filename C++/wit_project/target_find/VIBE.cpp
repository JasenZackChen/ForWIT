#include <opencv2/opencv.hpp>  
#include <iostream>  
#include "VIBE.h"  
#include "windows.h" 
using namespace std;  
int sig=0;
int c_xoff[9] = {-1,  0,  1, -1, 1, -1, 0, 1, 0}; 
int c_yoff[9] = {-1,  0,  1, -1, 1, -1, 0, 1, 0};  
bool fflag=1;  
ViBe_BGS::ViBe_BGS(void)  
{  
  
}  
ViBe_BGS::~ViBe_BGS(void)  
{  
  
}  
  
/**************** Assign space and init ***************************/  
void ViBe_BGS::init(const cv::Mat _image)  
{  
     for(int i = 0; i < NUM_SAMPLES; i++) 
     {  
         m_samples[i] = cv::Mat::zeros(_image.size(), CV_8UC1); 
     }  
     m_mask = cv::Mat::zeros(_image.size(),CV_8UC1); 
     m_foregroundMatchCount = cv::Mat::zeros(_image.size(),CV_8UC1);  
}  
  
/**************** Init model from first frame ********************/  
void ViBe_BGS::processFirstFrame(const cv::Mat _image)  
{  
    cv::RNG rng;									
    int row, col;  
  
    for(int i = 0; i < _image.rows; i++)  
    {  
        for(int j = 0; j < _image.cols; j++)  
        {  
             for(int k = 0 ; k < NUM_SAMPLES; k++)  
             {  
                 // Random pick up NUM_SAMPLES pixel in neighbourhood to construct the model  
                 int random = rng.uniform(0,9);  
  
                 row = i + c_yoff[random];  
                 if (row < 0)   
                     row = 0;  
                 if (row >= _image.rows)  
                     row = _image.rows - 1;  
  
                 col = j + c_xoff[random];  
                 if (col < 0)    
                     col = 0;  
                 if (col >= _image.cols)  
                     col = _image.cols - 1;  
  
                 m_samples[k].at<uchar>(i, j) = _image.at<uchar>(row, col);  
             }  
        }  
    }  
}  
  
/**************** Test a new frame and update model ********************/  
void ViBe_BGS::testAndUpdate(const cv::Mat _image)  
{  
    cv::RNG rng;
    int c=0;  
  
    for(int i = 0; i < _image.rows; i++)  
    {  
        for(int j = 0; j < _image.cols; j++)  
        {  
            int matches(0), count(0);  
            float dist;  
  
            while(matches < MIN_MATCHES && count < NUM_SAMPLES) 
            {  
                dist = abs(m_samples[count].at<uchar>(i, j) - _image.at<uchar>(i, j)); 
                if (dist < RADIUS)  
                    matches++; 
 
                count++;  
            }  
  
            if (matches >= MIN_MATCHES)  
            {  
                // It is a background pixel  
                m_foregroundMatchCount.at<uchar>(i, j) = 0;  
  
                // Set background pixel to 0  
                m_mask.at<uchar>(i, j) = 0; 
  
               
                int random = rng.uniform(0, SUBSAMPLE_FACTOR); 
                if (random == 0)  
                {  
                    random = rng.uniform(0, NUM_SAMPLES);  
                    m_samples[random].at<uchar>(i, j) = _image.at<uchar>(i, j);  
                }  
  
                 
                random = rng.uniform(0, SUBSAMPLE_FACTOR);  
                if (random == 0)  
                {  
                    int row, col;  
                    random = rng.uniform(0, 9);  
                    row = i + c_yoff[random];  
                    if (row < 0)   
                        row = 0;  
                    if (row >= _image.rows)  
                        row = _image.rows - 1;  
  
                    random = rng.uniform(0, 9);  
                    col = j + c_xoff[random];  
                    if (col < 0)   
                        col = 0;  
                    if (col >= _image.cols)  
                        col = _image.cols - 1;  
  
                    random = rng.uniform(0, NUM_SAMPLES);  
                    m_samples[random].at<uchar>(row, col) = _image.at<uchar>(i, j);  
                }  
            } 
 
            else  
            {  
                // It is a foreground pixel  
                m_foregroundMatchCount.at<uchar>(i, j)++;  
   
                // Set background pixel to 255  
                m_mask.at<uchar>(i, j) =255;  
  
                if (m_foregroundMatchCount.at<uchar>(i, j) > 50)  
                {  
                    
                    c++;
                    // if(sig>0)
                    // sig=-1;
                    // return 1;
                    int random = rng.uniform(0, SUBSAMPLE_FACTOR);  
                    if (random == 0)  
                    {  
                        random = rng.uniform(0, NUM_SAMPLES);  
                        m_samples[random].at<uchar>(i, j) = _image.at<uchar>(i, j);  
                    }  
                }
                if (m_foregroundMatchCount.at<uchar>(i, j) > 90)  
                {  
                    c=0;
                    // if(sig>0)
                    // sig=-1;
                    // return 1;
                    // int random = rng.uniform(0, SUBSAMPLE_FACTOR);  
                    // if (random == 0)  
                    // {  
                    //     random = rng.uniform(0, NUM_SAMPLES);  
                    //     m_samples[random].at<uchar>(i, j) = _image.at<uchar>(i, j);  
                    // }  
                }    
            }  
        }  
    }
    if(c>4000&&fflag)
    {
        cout<<"error"<<endl; 
        sig=1;
        fflag=0;
    }
    // return 0;
}  