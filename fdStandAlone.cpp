#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace cv::dnn;

#include <iostream>
#include <cstdlib>
using namespace std;

const size_t inWidth = 300;
const size_t inHeight = 300;
const double inScaleFactor = 1.0;
const Scalar meanVal(104.0, 177.0, 123.0);

dnn::Net init_net_caffe(std::string proto, std::string model)
{
    std::cout<< "init_net_caffe started..." << std::endl;

    dnn::Net net;
    net = readNetFromCaffe(proto, model);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

    if (net.empty())
    {
        std::cerr << "init_net_caffe failed: " << std::endl;
        std::cerr << "prototxt:   " << proto << std::endl;
        std::cerr << "caffemodel: " << model << std::endl;
    }

    return net;
}

std::vector<Rect> face_detection(dnn::Net net, Mat frame, float confidenceThreshold)
{
    std::vector<Rect> rectList;

    if (frame.empty())
    {
        std::cout << "Frame is empty.";
        return rectList;
    }

    if (frame.channels() == 4)
        cvtColor(frame, frame, COLOR_BGRA2BGR);

    Mat inputBlob = blobFromImage(frame, 1.0, Size(300, 300), (104.0, 177.0, 123.0), false, false);

    net.setInput(inputBlob, "data");

    Mat detection = net.forward("detection_out");

    Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

    for(int i = 0; i < detectionMat.rows; i++)
    {
        float confidence = detectionMat.at<float>(i, 2);

        if(confidence > confidenceThreshold)
        {
            int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
            int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
            int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
            int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);

            std::cout << "push_back x: " << xLeftBottom << "y: " << yLeftBottom << "width: " << xRightTop << "height: " << yRightTop << std::endl;

            Rect object((int)xLeftBottom, (int)yLeftBottom, (int)(xRightTop - xLeftBottom), (int)(yRightTop - yLeftBottom));
            rectList.push_back(object);
        }
    }

    return rectList;
}

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


