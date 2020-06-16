#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <assert.h>

using namespace cv;
using namespace cv::dnn;

#include <iostream>
#include <cstdlib>
using namespace std;

const size_t inWidth = 300;
const size_t inHeight = 300;
const double inScaleFactor = 1.0;
const Scalar meanVal(104.0, 177.0, 123.0);

const char* about = "This sample uses Single-Shot Detector "
                    "(https://arxiv.org/abs/1512.02325) "
                    "with ResNet-10 architecture to detect faces on camera/video/image.\n"
                    "More information about the training is available here: "
                    "<OPENCV_SRC_DIR>/samples/dnn/face_detector/how_to_train_face_detector.txt\n"
                    ".caffemodel model's file is available here: "
                    "<OPENCV_SRC_DIR>/samples/dnn/face_detector/res10_300x300_ssd_iter_140000.caffemodel\n"
                    ".prototxt file is available here: "
                    "<OPENCV_SRC_DIR>/samples/dnn/face_detector/deploy.prototxt\n";

const char* params
    = "{ help           | false | print usage          }"
      "{ proto          | deploy.prototxt      | model configuration (deploy.prototxt) }"
      "{ model          | res10_300x300_ssd_iter_140000.caffemodel     | model weights (res10_300x300_ssd_iter_140000.caffemodel) }"
      "{ camera_device  | 0     | camera device number }"
      "{ video          |       | video or image for detection }"
      "{ min_confidence | 0.5   | min confidence       }";

int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv, params);

    if (parser.get<bool>("help"))
    {
        cout << about << endl;
        parser.printMessage();
        return 0;
    }

    String modelConfiguration = parser.get<string>("proto");
    String modelBinary = parser.get<string>("model");

    //! [Initialize network]
    dnn::Net net = readNetFromCaffe(modelConfiguration, modelBinary);
    // 若为tensorflow模型，则使用readNetFromTensorflow，需要用到.pbtxt格式的配置文件和.pb格式的模型文件
    // dnn::Net net = readNetFromTensorflow(modelBinary, modelConfiguration);
    //! [Initialize network]

    if (net.empty())
    {
        cerr << "Can't load network by using the following files: " << endl;
        cerr << "prototxt:   " << modelConfiguration << endl;
        cerr << "caffemodel: " << modelBinary << endl;
        cerr << "Models are available here:" << endl;
        cerr << "<OPENCV_SRC_DIR>/samples/dnn/face_detector" << endl;
        cerr << "or here:" << endl;
        cerr << "https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector" << endl;
        exit(-1);
    }

   net.setPreferableBackend(DNN_BACKEND_CUDA);
   net.setPreferableTarget(DNN_TARGET_CUDA);

    VideoCapture cap;
    if (parser.get<String>("video").empty())
    {
        int cameraDevice = parser.get<int>("camera_device");
        cap = VideoCapture(cameraDevice);
        if(!cap.isOpened())
        {
            cout << "Couldn't find camera: " << cameraDevice << endl;
            return -1;
        }
    }
    else
    {
        cap.open(parser.get<String>("video"));
        if(!cap.isOpened())
        {
            cout << "Couldn't open image or video: " << parser.get<String>("video") << endl;
            return -1;
        }
    }

    for(;;)
    {
        Mat image;
        cap >> image; // get a new frame from camera/video or read image

        if (image.empty())
        {
            waitKey();
            break;
        }

        cv::Mat image_result = image.clone();

        if (image.channels() == 4)
            cvtColor(image, image, COLOR_BGRA2BGR);

        //! [Prepare blob]
        //!  image: 3 channels
        Mat inputBlob = blobFromImage(image, inScaleFactor,
                                      Size(image.cols, image.rows), meanVal, false, false); //Convert Mat to batch of images
        //! [Prepare blob]

        //! [Set input blob]
        net.setInput(inputBlob, "data"); //set the network input
        //! [Set input blob]

        //! [Make forward pass]
        Mat detection = net.forward("detection_out"); //compute output
        //! [Make forward pass]

        vector<double> layersTimings;
        double freq = getTickFrequency() / 1000;
        double time = net.getPerfProfile(layersTimings) / freq;

        Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

        ostringstream ss;
        ss << "FPS: " << 1000/time << " ; time: " << time << " ms";
        putText(image_result, ss.str(), Point(20,20), 0, 0.5, Scalar(0,0,255));

        float confidenceThreshold = parser.get<float>("min_confidence");
        for(int i = 0; i < detectionMat.rows; i++)
        {
            float confidence = detectionMat.at<float>(i, 2);

            if(confidence > confidenceThreshold)
            {
                int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * image.cols);
                int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * image.rows);
                int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * image.cols);
                int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * image.rows);

                Rect object((int)xLeftBottom, (int)yLeftBottom,
                            (int)(xRightTop - xLeftBottom),
                            (int)(yRightTop - yLeftBottom));

                rectangle(image_result, object, Scalar(0, 255, 0));

                ss.str("");
                ss << confidence;
                String conf(ss.str());
                String label = "Face: " + conf;
                int baseLine = 0;
                Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
                //rectangle(image_result, Rect(Point(xLeftBottom, yLeftBottom - labelSize.height), Size(labelSize.width, labelSize.height + baseLine)), Scalar(255, 255, 255), CV_FILLED);
                rectangle(image_result, Rect(Point(xLeftBottom, yLeftBottom - labelSize.height), Size(labelSize.width, labelSize.height + baseLine)), Scalar(255, 255, 0), 1);
                putText(image_result, label, Point(xLeftBottom, yLeftBottom), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255,0,255));
            }
        }

        imshow("detections", image_result);
        if (waitKey(1) >= 0) break;
    }

    return 0;
} // main
