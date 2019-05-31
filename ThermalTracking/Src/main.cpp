/*
 * main.cpp
 *
 *  Created on: 23 de mar de 2019
 *      Author: Julhio Cesar Navas
 */

#include "../Inc/main.h"

using namespace std;
using namespace cv;

Mat infrared_cam;

float ktoc(float val){
	return ((val - 27315) / 100.0);
}
//uint8_t channels = 0;
Mat raw_to_8bit(Mat frame){
	Mat colorMat16;

	//channels = frame.channels();

	normalize(frame, frame, 0, 65535, NORM_MINMAX, CV_16UC1);

	Mat image_grayscale = frame.clone();
	image_grayscale.convertTo(image_grayscale, CV_8UC1, 1 / 256.0);
	if(frame.size){
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


void display_temperature(Mat img, double val_k, Point loc, Scalar color){
	float val = ktoc(val_k);
	char text[10];
	sprintf(text,"%.2f degC", val);
	putText(img,text, Point(loc), FONT_HERSHEY_SIMPLEX, 0.75, color, 2);
	int x, y;
	x = loc.x;
	y = loc.y;
	line(img, Point(x - 2, y), Point(x + 2, y), color, 1, LINE_4);
	line(img, Point(x, y - 2), Point(x, y + 2), color, 1, LINE_4);
}

CascadeClassifier face_cascade;
String window_name = "Face Tracking";

/**
 * Detects faces and draws an ellipse around them
 */
Mat detectFaces(Mat frame) {

	std::vector<Rect> faces;
	Mat frame_gray;

	// Convert to gray scale
	if(frame.size){
		cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	}
	// Equalize histogram
	equalizeHist(frame_gray, frame_gray);

	// Detect faces
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 3,
			0|CASCADE_SCALE_IMAGE, Size(30, 30));

	// Iterate over all of the faces
	for( size_t i = 0; i < faces.size(); i++ ) {

		// Find center of faces
		Point center(faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2);

		rectangle(frame, faces[i], cv::Scalar(0,255,0), LINE_4);
	}

	return frame;
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

	if(!visible_cam.open(0)){ // Open default camera
		visible_cam.open(1);
	}
	//Enable and disable the autofocus
	visible_cam.set(CAP_PROP_AUTOFOCUS, 1);
	visible_cam.set(CAP_PROP_AUTOFOCUS, 0);
	//Mat frame;
	visible_cam.set(3,640);
	visible_cam.set(4,480);

	// Load preconstructed classifier
	face_cascade.load("Data/haarcascade_frontalface_alt.xml");

	cv::Mat3b visible_frame;
	Mat visible_frame_out;
	while(true){

		visible_cam >> visible_frame;

		namedWindow(window_name, cv::WINDOW_NORMAL);
		resizeWindow(window_name, 1280,480);

		try{
			if(!uvcacquisition->returnQueue().empty()) {
				infrared_data = uvcacquisition->returnQueue().back();
			}
			else{
				//break;
			}

			resize(infrared_data, infrared_data, Size(640,480), 0, 0, INTER_NEAREST);
			minMaxLoc(infrared_data, &minVal, &maxVal, &minLoc, &maxLoc);
			infrared_cam = raw_to_8bit(infrared_data);
			display_temperature(infrared_cam, minVal, minLoc, Scalar(255, 0, 0));
			display_temperature(infrared_cam, maxVal, maxLoc, Scalar(0, 0, 255));

			// Display frame
			if (!infrared_cam.empty()) {

				// Create 1280x480 mat for window
				cv::Mat win_mat(cv::Size(1280, 480), CV_8UC3);

				// Copy small images into big mat
				detectFaces(visible_frame).copyTo(win_mat(cv::Rect(  0, 0, 640, 480)));
				infrared_cam.copyTo(win_mat(cv::Rect(640, 0, 640, 480)));

				// Display big mat
				imshow(window_name, win_mat);
			}
		}
		catch (exception& e){
			cout << "Standard exception: " << e.what() << endl;
		}

		// Press  ESC on keyboard to exit
		char c = waitKey(1);
		if(c==27)
			break;
	}

	destroyAllWindows();

	uvcacquisition->pauseStream();

	return 0;
}

//----------------------------------------------------------------------------------
