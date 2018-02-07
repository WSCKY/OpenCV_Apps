#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>

using namespace std;
using namespace cv;

int main(void)
{
	vector<String> fileName;

	fileName.push_back("../Resources/rock_s_60_r.jpg");
	fileName.push_back("../Resources/rock_s.jpg");

	Mat img1 = imread(fileName[0], IMREAD_GRAYSCALE);
	Mat img2 = imread(fileName[1], IMREAD_GRAYSCALE);

	if (img1.rows*img1.cols <= 0) {
		cout << "Image " << fileName[0] << " is empty or cannot be found\n";
		return(0);
	}
	if (img2.rows*img2.cols <= 0) {
		cout << "Image " << fileName[1] << " is empty or cannot be found\n";
		return(0);
	}
vector<String> typeAlgoMatch;

typeAlgoMatch.push_back("BruteForce-Hamming(2)");
typeAlgoMatch.push_back("BruteForce");
typeAlgoMatch.push_back("BruteForce-L1");
typeAlgoMatch.push_back("BruteForce-Hamming");

	Ptr<Feature2D> b = BRISK::create();

	// Match method
	Ptr<DescriptorMatcher> descriptorMatcher;
	// Match between img1 and img2
	vector<DMatch> matches;
	// keypoint  for img1 and img2
	vector<KeyPoint> keyImg1, keyImg2;
	// Descriptor for img1 and img2
	Mat descImg1, descImg2;
	vector<String>::iterator itMatcher = typeAlgoMatch.begin();
try {
	// We can detect keypoint with detect method
	b->detect(img1, keyImg1, Mat());
	// and compute their descriptors with method  compute
	b->compute(img1, keyImg1, descImg1);
	// or detect and compute descriptors in one step
	b->detectAndCompute(img2, Mat(),keyImg2, descImg2,false);

	descriptorMatcher = DescriptorMatcher::create(*itMatcher);
	descriptorMatcher->match(descImg1, descImg2, matches, Mat());
	// Keep best matches only to have a nice drawing.
	// We sort distance between descriptor matches
	Mat index;
	int nbMatch=int(matches.size());
	Mat tab(nbMatch, 1, CV_32F);
	for (int i = 0; i < nbMatch; i ++) {
		tab.at<float>(i, 0) = matches[i].distance;
	}
	sortIdx(tab, index, SORT_EVERY_COLUMN + SORT_ASCENDING);
	vector<DMatch> bestMatches;
float scal = 0.0f;
float l1, l2;
float x1, y1, x2, y2;
	for (int i = 0; i < 20; i ++) {//nbMatch
		bestMatches.push_back(matches[index.at<int>(i, 0)]);
x1 = keyImg1[bestMatches[i].queryIdx].pt.x;
y1 = keyImg1[bestMatches[i].queryIdx].pt.y;
x2 = keyImg2[bestMatches[i].trainIdx].pt.x;
y2 = keyImg2[bestMatches[i].trainIdx].pt.y;
l1 = sqrtf(x1 * x1 + y1 * y1);
l2 = sqrtf(x2 * x2 + y2 * y2);
cout << i << ": " << keyImg1[bestMatches[i].queryIdx].pt.x << ", " << keyImg1[bestMatches[i].queryIdx].pt.y << "\t";
cout << " -> " << keyImg2[bestMatches[i].trainIdx].pt.x << ", " << keyImg2[bestMatches[i].trainIdx].pt.y << "\t";
cout << "scale: " << l1/l2 << endl;
scal += l1/l2;
	}
scal /= 20;
cout << "scale average = " << scal << endl;
	Mat result;
	drawMatches(img1, keyImg1, img2, keyImg2, bestMatches, result);
	namedWindow("BRISK: "+*itMatcher, WINDOW_AUTOSIZE);
	imshow("BRISK: " + *itMatcher, result);
	waitKey();
} catch (Exception& e) {
	cout << "Feature : " << "BRISK" << "\n";
	cout << "Matcher : " << *itMatcher << "\n";
	cout << e.msg << endl;
}
	return 0;
}

