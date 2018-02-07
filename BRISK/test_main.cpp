#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
	//Load Image
	Mat c_src1 =  imread( "../Resources/r_b.jpg");
	Mat c_src2 = imread("../Resources/r_s.jpg");
	Mat src1 = imread( "../Resources/r_b.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	Mat src2 = imread( "../Resources/r_s.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	if( !src1.data || !src2.data )
	{
		cout<< "Error reading images " << std::endl;
		return -1;
	}
	//feature detect
	Ptr<BRISK> detector = BRISK::create();
	vector<KeyPoint> kp1, kp2;
	int64 start = getTickCount();
	detector->detect( src1, kp1 );
	detector->detect( src2, kp2 );
	//cv::BRISK extractor;
	Mat des1,des2;//descriptor
	detector->compute(src1, kp1, des1);
	detector->compute(src2, kp2, des2);
	Mat res1,res2;
	int drawmode = DrawMatchesFlags::DRAW_RICH_KEYPOINTS;
	drawKeypoints(c_src1, kp1, res1, Scalar::all(-1), drawmode);//画出特征点
	drawKeypoints(c_src2, kp2, res2, Scalar::all(-1), drawmode);
	cout<<"size of description of Img1: "<<kp1.size()<<endl;
	cout<<"size of description of Img2: "<<kp2.size()<<endl;

	BFMatcher matcher(NORM_HAMMING);
	vector<DMatch> matches;
	matcher.match(des1, des2, matches);
	int64 end = getTickCount();
	cout<<"耗时："<<(end - start) / 1000000 <<"ms"<<endl;
	Mat img_match;
	drawMatches(src1, kp1, src2, kp2, matches, img_match);
	cout<<"number of matched points: "<<matches.size()<<endl;
	imshow("matches",img_match);
	cvWaitKey(0);
	cvDestroyAllWindows();
	return 0;
}

