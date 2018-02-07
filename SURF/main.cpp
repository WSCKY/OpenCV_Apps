#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include "opencv2/videoio.hpp"

#include "opencv2/xfeatures2d.hpp"

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

int main(void)
{
	vector<String> fileName;

	fileName.push_back("../Resources/off_cut_scal.jpg");
	fileName.push_back("../Resources/off_b_s.jpg");

	Mat img1 = imread(fileName[0], IMREAD_GRAYSCALE);
	Mat img2 = imread(fileName[1], IMREAD_GRAYSCALE);

	Ptr<SURF> surf;
	surf = SURF::create(800);

	BFMatcher matcher;
	// Descriptor for img1 and img2
	Mat descImg1, descImg2;
	// keypoint  for img1 and img2
	vector<KeyPoint> keyImg1, keyImg2;
	// Match between img1 and img2
	vector<DMatch> matches;

	surf->detectAndCompute(img1, Mat(), keyImg1, descImg1);
	surf->detectAndCompute(img2, Mat(), keyImg2, descImg2);

	matcher.match(descImg1, descImg2, matches);

	sort(matches.begin(), matches.end());

	vector<DMatch> bestMatches;
	int ptsPairs = std::min(50, (int)(matches.size() * 0.15));
	cout << ptsPairs << endl;
	for (int i = 0; i < ptsPairs; i++) {
		bestMatches.push_back(matches[i]);
	}

Mat result;  
    drawMatches(img1, keyImg1, img2, keyImg2, bestMatches, result);//, Scalar::all(-1), Scalar::all(-1),vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS

std::vector<Point2f> obj;
    std::vector<Point2f> scene;
  
    for (size_t i = 0; i < bestMatches.size(); i++)   {
        obj.push_back(keyImg1[bestMatches[i].queryIdx].pt);
        scene.push_back(keyImg2[bestMatches[i].trainIdx].pt);
    }

    std::vector<Point2f> obj_corners(4);
    obj_corners[0] = Point(0, 0);
    obj_corners[1] = Point(img1.cols, 0);
    obj_corners[2] = Point(img1.cols, img1.rows);
    obj_corners[3] = Point(0, img1.rows);
    std::vector<Point2f> scene_corners(4);

    Mat H = findHomography(obj, scene, RANSAC);
    perspectiveTransform(obj_corners, scene_corners, H);  

    line(result,scene_corners[0] + Point2f((float)img1.cols, 0), scene_corners[1] + Point2f((float)img1.cols, 0),Scalar(0, 255, 0), 2, LINE_AA);
    line(result,scene_corners[1] + Point2f((float)img1.cols, 0), scene_corners[2] + Point2f((float)img1.cols, 0),Scalar(0, 255, 0), 2, LINE_AA);
    line(result,scene_corners[2] + Point2f((float)img1.cols, 0), scene_corners[3] + Point2f((float)img1.cols, 0),Scalar(0, 255, 0), 2, LINE_AA);
    line(result,scene_corners[3] + Point2f((float)img1.cols, 0), scene_corners[0] + Point2f((float)img1.cols, 0),Scalar(0, 255, 0), 2, LINE_AA);
    imshow("surf-compare", result);
    cvWaitKey(0);
}

