#include "fd.h"

using namespace cv;
using namespace cv::dnn;

#include <iostream>
#include <cstdlib>
using namespace std;

/*
 * g++ testLib.cpp -std=c++11 -I /home/xugaoxiang/anaconda3/include/opencv4/ -L /home/xugaoxiang/anaconda3/lib -lopencv_core -lopencv_dnn -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_videoio libfacedetection.so
 * */

int main(int argc, char** argv)
{
    dnn::Net net = init_net_caffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel");

    if (net.empty())
    {
        cerr << "Can't load network by using the following files: " << endl;
        cerr << "Models are available here:" << endl;
        cerr << "<OPENCV_SRC_DIR>/samples/dnn/face_detector" << endl;
        cerr << "or here:" << endl;
        cerr << "https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector" << endl;
        exit(-1);
    }

    VideoCapture cap;

    // using build-in camera to test
    cap = VideoCapture(0);
    if(!cap.isOpened())
    {
        cout << "Couldn't find camera: " << endl;
        return -1;
    }

    for(;;)
    {
        Mat frame;
        cap >> frame; // get a new frame from camera/video or read image

        if (frame.empty())
        {
            waitKey();
            break;
        }

        if (frame.channels() == 4)
            cvtColor(frame, frame, COLOR_BGRA2BGR);

        vector<Rect>rectList;
        rectList = face_detection(net, frame, 0.5);

        for(size_t j = 0; j < rectList.size(); j++)
        {
            Rect r = rectList[j];
            cout << "begin x=" << r.x << "y=" << r.y << "width=" << r.width << "height=" << r.height << endl;
        }

        imshow("detections", frame);
        if (waitKey(1) >= 0) break;
    }

    return 0;
}

