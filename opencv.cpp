#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;

int main()
{
    cv::Mat img;
    img = cv::imread("../img/1.jpg");
    if (img.empty())
    {
        cerr << "Image load failed." << endl;
        return 1;
    }

    cv::namedWindow("image");
    cv::imshow("image", img);
    cv::waitKey(0);
    return 0;
}