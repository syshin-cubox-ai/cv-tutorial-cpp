#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;

int main()
{
    cv::VideoCapture cap("../img/1.mp4");
    if (!cap.isOpened())
    {
        cerr << "Camera open failed." << endl;
        return -1;
    }

    cv::Mat img;
    while (cv::waitKey(20) != int('q'))
    {
        bool ret = cap.read(img);
        if (!ret)
        {
            cerr << "No frame has been grabbed." << endl;
            return 0;
        }

        cv::imshow("Face Detection", img);
    }

    cout << "Quit inference." << endl;
    cap.release();
    cv::destroyAllWindows();
    return 0;
}