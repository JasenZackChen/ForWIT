#pragma once  
#include <iostream>  
#include "opencv2/opencv.hpp" 
  
using namespace std;  
  
#define NUM_SAMPLES 20      //ÿ�����ص����������  
#define MIN_MATCHES 2       //#minָ��  
#define RADIUS 20       //Sqthere�뾶  
#define SUBSAMPLE_FACTOR 16 //�Ӳ������ʣ������������µĸ���
  
  
class ViBe_BGS  
{  
public:  
    ViBe_BGS(void);  //���캯��
    ~ViBe_BGS(void);  //�����������Կ��ٵ��ڴ�����Ҫ��������
  
    void init(const cv::Mat _image);   //��ʼ��  
    void processFirstFrame(const cv::Mat _image); //���õ�һ֡���н�ģ 
    void testAndUpdate(const cv::Mat _image);  //�ж�ǰ���뱳���������б������� 
    cv::Mat getMask(void){return m_mask;};  //�õ�ǰ��
  
private:  
    cv::Mat m_samples[NUM_SAMPLES];  //ÿһ֡ͼ���ÿһ�����ص�������
    cv::Mat m_foregroundMatchCount;  //ͳ�����ر��ж�Ϊǰ���Ĵ��������ڸ���
    cv::Mat m_mask;  //ǰ����ȡ���һ֡ͼ��
};  