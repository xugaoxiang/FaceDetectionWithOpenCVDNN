//
// face detection dynamic library using opencv dnn module
//

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace cv::dnn;

#include <iostream>
#include <cstdlib>
#include <vector>

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
    std::cout << "face_detection started..." << std::endl;
    std::vector<Rect> rectList;

    if (frame.empty())
    {
        std::cout << "Frame is empty.";
        return rectList;
    }

    if (frame.channels() == 4)
        cvtColor(frame, frame, COLOR_BGRA2BGR);

    Mat inputBlob = blobFromImage(frame, 1.0, Size(frame.cols, frame.rows), (104.0, 177.0, 123.0), false, false);

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

            Rect object((int)xLeftBottom, (int)yLeftBottom, (int)(xRightTop - xLeftBottom), (int)(yRightTop - yLeftBottom));
            rectList.push_back(object);
        }
    }

    return rectList;
}
