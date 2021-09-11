/*
 * main.cpp
 *
 *  Created on: 23 de mar de 2019
 *      Author: Julhio Cesar Navas
 */

// https://github.com/bm777/humanface-mask-detector


#include "../Inc/main.h"

using namespace std;
using namespace cv;
using namespace cv::face;

#define WEBCAM_USB_VID		0x0c45
#define WEBCAM_USB_PID		0x6a08

Mat infrared_cam;

float ktoc(float val) {
	return ((val - 27315) / 100.0);
}
//uint8_t channels = 0;
Mat raw_to_8bit(Mat frame) {
	Mat colorMat16;

	//channels = frame.channels();

	normalize(frame, frame, 0, 65535, NORM_MINMAX, CV_16UC1);

	Mat image_grayscale = frame.clone();
	image_grayscale.convertTo(image_grayscale, CV_8UC1, 1 / 256.0);
	if (frame.size) {
		cvtColor(image_grayscale, colorMat16, COLOR_GRAY2RGB);
	}
	return colorMat16;
}

/*Mat norm_0_255(Mat src) {

 // Create and return normalized image:
 Mat dst;

 switch(src.channels()) {

 case 1:

 normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);

 break;

 case 3:

 cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC3);

 break;

 default:

 src.copyTo(dst);

 break;

 }

 return dst;

 }*/

void display_temperature(Mat img, double val_k, Point loc, Scalar color) {
	float val = ktoc(val_k);
	char text[10];
	sprintf(text, "%.2f degC", val);
	putText(img, text, Point(loc), FONT_HERSHEY_SIMPLEX, 0.75, color, 2);
	int x, y;
	x = loc.x;
	y = loc.y;
	line(img, Point(x - 2, y), Point(x + 2, y), color, 1, LINE_4);
	line(img, Point(x, y - 2), Point(x, y + 2), color, 1, LINE_4);
}

CascadeClassifier face_cascade;
String window_name = "Face Tracking";

// Face Alignment
void faceAlignment(const Mat& img, Mat& faceImgAligned, float* eyeCenters, float* eyeCenters_ref, Size faceSize){
	float dist_ref = eyeCenters_ref[2] - eyeCenters_ref[0];
    float dx = eyeCenters[2] - eyeCenters[0];
    float dy = eyeCenters[3] - eyeCenters[1];
    float dist = sqrt(dx * dx + dy * dy);

    // scale
    double scale = dist_ref / dist;
    // angle
    double angle = atan2(dy,dx) * 180 / M_PI;
    // center
    cv::Point center = cv::Point(0.5 * (eyeCenters[0] + eyeCenters[2]), 0.5 * (eyeCenters[1] + eyeCenters[3]));

    // Calculat rotation matrix
    Mat rot = getRotationMatrix2D(center, angle, scale);

    rot.at<double>(0, 2) += faceSize.width * 0.5 - center.x;
    rot.at<double>(1, 2) += eyeCenters[1] - center.y;

    // Apply affine transform
    cv::Mat imgIn = img.clone();
    imgIn.convertTo(imgIn, CV_32FC3, 1. / 255.);
    warpAffine(imgIn, faceImgAligned, rot, faceSize);
    faceImgAligned.convertTo(faceImgAligned, CV_8UC3, 255);
}

/**
 * Detects faces and draws an ellipse around them
 */
Mat detectFaces(Mat frame) {

	std::vector<Rect> faces;
	Mat frame_gray;

	// Convert to gray scale
	if (frame.size) {
		cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	}
	// Equalize histogram
	equalizeHist(frame_gray, frame_gray);

	// Detect faces
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 3,
			0 | CASCADE_SCALE_IMAGE, Size(30, 30));

	// Iterate over all of the faces
	for (size_t i = 0; i < faces.size(); i++) {

		// Find center of faces
		cv::Point center(faces[i].x + faces[i].width / 2,
				faces[i].y + faces[i].height / 2);

		rectangle(frame, faces[i], cv::Scalar(0, 255, 0), LINE_4);
	}

	return frame;
}

using rectPoints = std::pair<cv::Rect, std::vector<cv::Point>>;

static cv::Mat drawRectsAndPoints(const cv::Mat &img, const std::vector<rectPoints> data) {
	cv::Mat outImg;
	img.convertTo(outImg, CV_8UC3);

	for (auto &d : data) {
		cv::rectangle(outImg, d.first, cv::Scalar(0, 0, 255));
		auto pts = d.second;
		for (size_t i = 0; i < pts.size(); ++i) {
			cv::circle(outImg, pts[i], 3, cv::Scalar(0, 0, 255));
		}
	}
	return outImg;
}

Mat getFaceBoxMTCNN(Mat &frame){
	Mat frameOpenCV = frame.clone();

	ProposalNetwork::Config pConfig;
	pConfig.caffeModel = "Model/MTCNN/det1.caffemodel";
	pConfig.protoText =  "Model/MTCNN/det1.prototxt";
	pConfig.threshold = 0.6f;

	RefineNetwork::Config rConfig;
	rConfig.caffeModel = "Model/MTCNN/det2.caffemodel";
	rConfig.protoText = "Model/MTCNN/det2.prototxt";
	rConfig.threshold = 0.7f;

	OutputNetwork::Config oConfig;
	oConfig.caffeModel = "Model/MTCNN/det3.caffemodel";
	oConfig.protoText = "Model/MTCNN/det3.prototxt";
	oConfig.threshold = 0.7f;

	MTCNNDetector detector(pConfig, rConfig, oConfig);

	std::vector<Face> faces;
	faces = detector.detect(frameOpenCV, 50.f, 0.709f);

	std::vector<rectPoints> data;

	// show the image with faces in it
	for (size_t i = 0; i < faces.size(); ++i) {
		std::vector<cv::Point> pts;
		for (int p = 0; p < NUM_PTS; ++p) {
			pts.push_back(
			cv::Point(faces[i].ptsCoords[2 * p], faces[i].ptsCoords[2 * p + 1]));
		}

		auto rect = faces[i].bbox.getRect();
		auto d = std::make_pair(rect, pts);
		data.push_back(d);
	}

	auto resultImg = drawRectsAndPoints(frameOpenCV, data);

	return resultImg;
}

int main(int argc, char **argv) {
	Mat infrared_data;
	Point minLoc, maxLoc;
	double minVal, maxVal;

	//Constructor
	UvcAcquisition *uvcacquisition = new UvcAcquisition();
	// Start the video stream. The library will call user function callback:
	uvcacquisition->startStream();

	VideoCapture visible_cam;


	if (!visible_cam.open(0)) { // Open default camera
		visible_cam.open(1);
	}

	//Enable and disable the autofocus
	visible_cam.set(CAP_PROP_AUTOFOCUS, 1);
	visible_cam.set(CAP_PROP_AUTOFOCUS, 0);
	//Mat frame;
	visible_cam.set(CV_CAP_PROP_FRAME_WIDTH, 640);
	visible_cam.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

	// Load preconstructed classifier
	//face_cascade.load("Model/Haar/haarcascade_frontalface_alt.xml");


	ProposalNetwork::Config pConfig;
	pConfig.caffeModel = "Model/MTCNN/det1.caffemodel";
	pConfig.protoText =  "Model/MTCNN/det1.prototxt";
	pConfig.threshold = 0.6f;

	RefineNetwork::Config rConfig;
	rConfig.caffeModel = "Model/MTCNN/det2.caffemodel";
	rConfig.protoText = "Model/MTCNN/det2.prototxt";
	rConfig.threshold = 0.7f;

	OutputNetwork::Config oConfig;
	oConfig.caffeModel = "Model/MTCNN/det3.caffemodel";
	oConfig.protoText = "Model/MTCNN/det3.prototxt";
	oConfig.threshold = 0.7f;

	MTCNNDetector detector(pConfig, rConfig, oConfig);


	cv::Mat3b visible_frame;
	Mat visible_frame_out;
	double tt_opencv = 0;
	double fpsOpencv = 0;
	std::vector<Face> faces;

	while (true) {
		double t = cv::getTickCount();
		visible_cam >> visible_frame;
		// Create 1280x480 mat for window
		cv::Mat win_mat(cv::Size(1280, 480), CV_8UC3);

		namedWindow(window_name, cv::WINDOW_NORMAL);
		resizeWindow(window_name, 1280, 480);

		// Display visible frame
		if (!visible_frame.empty()) {
			// Copy visible images into big mat
//			detectFaces(visible_frame).copyTo(
//					win_mat(cv::Rect(0, 0, 640, 480)));
			faces = detector.detect(visible_frame, 50.f, 0.709f);

			std::vector<rectPoints> data;

			// show the image with faces in it
			for (size_t i = 0; i < faces.size(); ++i) {
				std::vector<cv::Point> pts;
				for (int p = 0; p < NUM_PTS; ++p) {
					pts.push_back(
							cv::Point(faces[i].ptsCoords[2 * p],
									faces[i].ptsCoords[2 * p + 1]));
				}

				auto rect = faces[i].bbox.getRect();
				auto d = std::make_pair(rect, pts);
				data.push_back(d);

				Mat resultImg = drawRectsAndPoints(visible_frame, data);

				resultImg.copyTo(win_mat(cv::Rect(0, 0, 640, 480)));
			}
		}

		if(uvcacquisition->isConnected()){
			try {
				if (!uvcacquisition->returnQueue().empty()) {
					infrared_data = uvcacquisition->returnQueue().back();
				} else {
					//break;
				}

				resize(infrared_data, infrared_data, Size(640, 480), 0, 0,
						INTER_NEAREST);
				minMaxLoc(infrared_data, &minVal, &maxVal, &minLoc, &maxLoc);
				infrared_cam = raw_to_8bit(infrared_data);
				//GaussianBlur(infrared_cam, infrared_cam, Size(640,480), 0); //for image smoothing
				display_temperature(infrared_cam, minVal, minLoc,
						Scalar(255, 0, 0));
				display_temperature(infrared_cam, maxVal, maxLoc,
						Scalar(0, 0, 255));

				if (!infrared_cam.empty()) {
					// Display infrared frame
					infrared_cam.copyTo(win_mat(cv::Rect(640, 0, 640, 480)));
				}

			} catch (exception &e) {
				cout << "Standard exception: " << e.what() << endl;
			}
		}

		tt_opencv = ((double) cv::getTickCount() - t) / cv::getTickFrequency();
		fpsOpencv = 1 / tt_opencv;
		putText(win_mat,
				format("OpenCV Classification ; FPS = %.2f", fpsOpencv),
				Point(10, 50), FONT_HERSHEY_SIMPLEX, 1.4, Scalar(0, 0, 255), 4);
		// Display big mat
		imshow(window_name, win_mat);

		// Press  ESC on keyboard to exit
		char c = waitKey(1);
		if (c == 27)
			break;
	}

	destroyAllWindows();

	uvcacquisition->pauseStream();

	return 0;
}

//----------------------------------------------------------------------------------
