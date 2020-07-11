#include <opencv2/opencv.hpp>
#include <opencv2/videostab.hpp>
#include "VIBE.cpp" 
#include <iostream>  
#include <cstdio>
#include <math.h> 
#include <stdlib.h>
#include <time.h>
using namespace std;
#define MAX_POINT 5

bool drawing_box = false;  
bool gotBox =  false;
char ky;
bool got_roi = false;
float start; //计时器
//计时程序
 
float endt ;
 
float lastt;
cv::Mat frame_draw_roi;
cv::Point points_array[MAX_POINT];
cv::Rect ROI_RECT;
vector<cv::Point>  co_ordinates;


cv::Rect box;
cv::Point downPoint;
void  mouse_click(int event, int x, int y, int flags, void *param)
{
	static float min_dist = 8;
	static int count = 0;
	switch (event)
	{
	case CV_EVENT_LBUTTONDOWN:
	{
		/*
		switch (count)	// number of set Point 
		{
		case 0:
			cout << "Select top-right    point" << endl;
			break;
		case 1:
			cout << "Select bottom-right point" << endl;
			break;
		case 2:
			cout << "Select bottom-left  point" << endl << endl;
			break;
		default:
			break;
		}
		*/
		if (!got_roi) // you are not select ROI yet!
		{
			points_array[count] = cv::Point(x, y);
			double dist_e2s = sqrt((points_array[count].x - points_array[0].x)*(points_array[count].x - points_array[0].x)
				+ (points_array[count].y - points_array[0].y)*(points_array[count].y - points_array[0].y));
			// cout << "distance: " << dist_e2s << endl;
			if (count >= 1)
			{
				if (dist_e2s < min_dist)
				{
					points_array[count] = points_array[0];
					cv::line(frame_draw_roi, points_array[count - 1], points_array[0], cv::Scalar(0, 255, 0), 2, CV_AA);
				}
				else
				{
					cv::circle(frame_draw_roi, points_array[count], 2, cv::Scalar(0, 255, 0), 2);	//show points on image
					line(frame_draw_roi, points_array[count - 1], points_array[count], cv::Scalar(0, 255, 0), 2, CV_AA);
				}
			}
			else
			cv::circle(frame_draw_roi, points_array[count], 2, cv::Scalar(0, 255, 0), 2);	//show points on image
			// cv::imshow("camera", frame_draw_roi);
			count++;
			if (count == MAX_POINT || dist_e2s < min_dist && count != 1) // if select 4 point finished
			{
				cout << "ROI x & y points :" << endl;
				for (int i = 0; i < count; i++)
					cout << points_array[i] << endl;
				cout << endl << "ROI Saved You can continue with double press any keys except 'c' " << endl << "once press 'c' or 'C' to clear points and retry select ROI " << endl << endl;
				// ky = cv::waitKey(0) & 0xFF;

				// if (ky == 99 || ky == 67)  // c or C to clear
				// {
				// 	for (int i = 0; i < MAX_POINT; i++)
				// 		points_array[i] = cv::Point(0, 0);
				// 	// imshow("camera", frame_draw_roi);
				// 	count = 0;
				// 	cout << endl << endl << endl << "@---------------------	 Clear Points!	------------------@ " << endl << endl << endl;
				// }
				// else // user accept points & dosn't want to clear them
				{
					// float min_x = std::min(points_array[0].x, points_array[3].x);	//find rectangle for minimum ROI surround it!
					// float max_x = std::max(points_array[1].x, points_array[2].x);
					// float min_y = std::min(points_array[0].y, points_array[1].y);
					// float max_y = std::max(points_array[3].y, points_array[2].y);
					// float height_roi = max_y - min_y;
					// float width_roi = max_x - min_x;
					// ROI_RECT = cv::Rect(min_x, min_y, width_roi, height_roi);
					got_roi = true;
					for (int i = 0; i < count; i++)
						co_ordinates.push_back(points_array[i]);
					ROI_RECT = boundingRect(co_ordinates);
				}
			}


		}
		else { // if got_roi se true => select roi before
			cout << endl << "You Select ROI Before " << endl << "if you want to clear point press 'c' or double press other keys to continue" << endl << endl;
		}
		break;
	}
	}

}


int main(int argc, char* argv[])  
{   
    cv::Mat frame, gray, mask,fframe,sframe,fsframe;
    cv::Mat imageROI; 
    string str; 
    
	str=R"(C++\wit_project\target_find\test.mp4)";

    cv::VideoCapture capture(str);
    
    if (!capture.isOpened())  
    {  
        cout<<"No camera or video input!\n"<<endl;  
        return -1;  
    }  
  
    ViBe_BGS Vibe_Bgs; 
    int count = 0; 
    int boxcount=1;
    bool flag =false;
    capture>>frame;
    fframe=frame.clone();
    frame_draw_roi=frame.clone();
    cv::namedWindow("camera",1);
    cv::setMouseCallback("camera",mouse_click,NULL);
    
    while(!got_roi)
        {
            cv::imshow("camera",frame_draw_roi);
            if(cv::waitKey(50) == 'q')//---------很重要    
            break;
        }
		cv::destroyWindow("camera"); 
    // while(!gotBox)    
    //      {    
    //          fframe.copyTo(frame);    
    //          rectangle(frame,box,cv::Scalar(255,0,0),2);//画出感兴趣区域  
    //          imshow("video",frame);    
    //          if(cv::waitKey(50) == 'q')//---------很重要    
    //              break;    
    //      }
    // cv::setMouseCallback("camera",NULL,NULL); 
    cout<<"1"<<endl; 

    while (1)  
    {  
        count++;  
        capture >> frame;  
        if (frame.empty())  
            break;
			  

        cv::Mat cut=frame(cv::Rect(ROI_RECT.x, ROI_RECT.y, ROI_RECT.width, ROI_RECT.height));  
        imshow("video",frame); 
        cv::cvtColor(cut, gray, CV_RGB2GRAY); 
      
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
            // cv::imshow("diff",fsframe);
        }  

        cv::Mat cframe=gray.clone();
        vector<vector<cv::Point>>contours;
        
	    vector<cv::Vec4i> hierarchy;
	    cv::findContours(mask,contours,hierarchy,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
        int maxx=-1,maxy=-1,minx=99999,miny=99999;
        if(sig==1)
	    for(int k = 0; k < contours.size(); k++) 
	    { 	
	    if (cv::contourArea(contours[k])>5000)
            {
			start=GetTickCount();
            cv::drawContours(cframe,contours,k,cv::Scalar(0,255,0));
			// vector<cv::Point>  co_error_img;
			// for (int i = 0; i < contours[k].size(); i++)
			// co_error_img.push_back(contours[k][i]);
			cv::Rect ROI_error; 
			ROI_error = boundingRect(cv::Mat(contours[k]));
			cv::Mat error_img;
			error_img=cut(cv::Rect(ROI_error.x,ROI_error.y,ROI_error.width,ROI_error.height));
			time_t rawtime;
			time ( &rawtime );
			char tmp[256];
    		strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S", localtime(&rawtime));
			cv::putText(error_img , tmp, cv::Point(0.1*error_img.cols,0.1*error_img.rows), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 255,0 ), 2, 3);
			cv::imwrite("./114.jpg", error_img);
			sig=0;
            }
        }
		endt=GetTickCount();
		if((endt-start)/cvGetTickFrequency () * 1000000 >1)
		fflag=1;
        cv::imshow("input", cframe);   
  
        if ( cv::waitKey(30) >=0 )  
            break;  
    }  
  system("pause");
    return 0;  
}  