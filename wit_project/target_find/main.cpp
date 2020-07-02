#include <opencv2/opencv.hpp>
#include <opencv2/videostab.hpp>
#include "VIBE.cpp" 
#include <iostream>  
#include <cstdio>
#include <math.h> 
#include <stdlib.h>
using namespace std;

cv::Rect box;
bool gotBB = false;
bool stop=false;
cv::Mat model;
cv::Mat TimeMat;

int main(int argc, char* argv[])  
{   
    cv::Mat frame, gray, mask,fframe,sframe,fsframe; 
    string str; 
    
	str=R"(wit_project\target_find\test.mp4)";

    cv::VideoCapture capture(str);
    
    if (!capture.isOpened())  
    {  
        cout<<"No camera or video input!\n"<<endl;  
        return -1;  
    }  
  
    ViBe_BGS Vibe_Bgs; 
    int count = 0; 
    int boxcount=1;


    while (1)  
    {  
        count++;  
        capture >> frame;  
        if (frame.empty())  
            break;  
        cv::cvtColor(frame, gray, CV_RGB2GRAY); 
      
        if (count == 1)  
        {  
            Vibe_Bgs.init(gray);  
            Vibe_Bgs.processFirstFrame(gray);  
            fframe=gray.clone();
            cout<<" Training GMM complete!"<<endl;  
        }  
        else  
        {  
            Vibe_Bgs.testAndUpdate(gray);  
            mask = Vibe_Bgs.getMask();  
            cv::morphologyEx(mask, mask, cv::MORPH_OPEN, cv::Mat());  
            if(count==2)
            sframe=mask.clone();
            cv::imshow("mask", mask);  
        }
        if(count==2){
            fsframe=abs(sframe-fframe);
            cv::imshow("diff",fsframe);
        }  

        cv::Mat cframe=frame.clone();
        vector<vector<cv::Point>>contours;
        
	    vector<cv::Vec4i> hierarchy;
	    cv::findContours(mask,contours,hierarchy,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
        int maxx=-1,maxy=-1,minx=99999,miny=99999;
        if(sig==1)
	    for(int k = 0; k < contours.size(); k++) 
	    { 	
	    if (cv::contourArea(contours[k])>5000)
            {
        
            cv::drawContours(cframe,contours,k,cv::Scalar(0,255,0));
            }
        }
        
        cv::imshow("input", cframe);   
  
        if ( cv::waitKey(30) >=0 )  
            break;  
    }  
  system("pause");
    return 0;  
}  