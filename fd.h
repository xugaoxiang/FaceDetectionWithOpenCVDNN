//
// Created by xugaoxiang on 2019/12/13.
//

#ifndef FACEDETECTION_FD_H
#define FACEDETECTION_FD_H

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace cv::dnn;

#include <iostream>
#include <cstdlib>
#include <vector>

dnn::Net init_net_caffe(std::string proto, std::string model);
std::vector<Rect> face_detection(dnn::Net net, Mat frame, float confidenceThreshold);

#endif //FACEDETECTION_FD_H
