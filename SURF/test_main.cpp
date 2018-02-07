#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"

#include "opencv2/xfeatures2d.hpp"

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

//主函数
int main()
{
    //载入源图像
    Mat srcImage = imread("../Resources/me_org.jpg", IMREAD_GRAYSCALE);
    namedWindow("image[origin]",1);
    imshow("image[origin]",srcImage);

    //检测surf关键点、提取训练图像描述符
    vector<KeyPoint> keyPoint;
    Mat descriptor;
Ptr<SURF> featureDetector;
    featureDetector = SURF::create(80);
featureDetector->detectAndCompute(srcImage, Mat(), keyPoint, descriptor);
//    featureDetector->detect(grayImage,keyPoint);
//    SurfDescriptorExtractor featureExtractor;
//    featureDetector->compute(grayImage, keyPoint, descriptor);

    //创建基于FLANN的描述符匹配对象
    FlannBasedMatcher matcher;
    vector<Mat> desc_collection(1,descriptor);
    matcher.add(desc_collection);
    matcher.train();

    //创建视频对象、定义帧率
    VideoCapture capture;
    capture.open(0);
    unsigned int frameCount=0;//帧数

    //不断循环，直到q键被按下
    while(char(waitKey(1))!=27)
    {
        //参数设置
        int64 time0=getTickCount();
        Mat testImage,grayImage_test;
        capture >> testImage;//采集视频到testImage中
        if(testImage.empty())
            continue;
        //转换图像到灰度
        cvtColor(testImage, grayImage_test, COLOR_BGR2GRAY);
        //检测S关键点、提取测试图像描述符
        vector<KeyPoint> keyPoint_test;
        Mat descriptor_test;
featureDetector->detectAndCompute(grayImage_test, Mat(), keyPoint_test, descriptor_test);
//        featureDetector->detect(grayImage_test, keyPoint_test);
//        featureExtractor->compute(grayImage_test, keyPoint_test, descriptor_test);

        //匹配训练和测试描述符
        vector< vector<DMatch> > matches;
        matcher.knnMatch(descriptor_test, matches, 2);
        //根据劳氏算法，得到优秀的匹配点
        vector<DMatch> goodMatches;
        for(unsigned int i=0;i<matches.size();++i)
        {
            if(matches[i][0].distance<0.6*matches[i][1].distance)
                goodMatches.push_back(matches[i][0]);
        }
        //绘制匹配点并显示窗口
        Mat dstImage;
        drawMatches(testImage, keyPoint_test, srcImage, keyPoint, goodMatches, dstImage);
        namedWindow("image[match]",1);
        imshow("image[match]",dstImage);

        //输出帧率信息
        cout << "当前帧率为：" <<getTickFrequency()/(getTickCount()-time0) << endl;
    }
    return 0;
}

