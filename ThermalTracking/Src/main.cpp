/*
 * main.cpp
 *
 *  Created on: 23 de mar de 2019
 *      Author: Julhio Cesar Navas
 */

#include "../Inc/main.h"

using namespace std;
using namespace cv;

Mat img;

float ktoc(float val){
	return ((val - 27315) / 100.0);
}

Mat raw_to_8bit(Mat data){
	Mat colorMat16;

	normalize(data, data, 0, 65535, NORM_MINMAX, CV_16UC1);

	Mat image_grayscale = data.clone();
	image_grayscale.convertTo(image_grayscale, CV_8UC1, 1 / 256.0);

	cvtColor(image_grayscale, colorMat16, COLOR_GRAY2RGB);

	return colorMat16;
}

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

int main(int argc, char **argv) {
	Mat data;
	Point minLoc, maxLoc;
	double minVal, maxVal;

	//Constructor
	UvcAcquisition *uvcacquisition = new UvcAcquisition();
	// Start the video stream. The library will call user function callback:
	uvcacquisition->startStream();

	while(true){

		namedWindow("Lepton Radiometry", cv::WINDOW_NORMAL);
		resizeWindow("Lepton Radiometry", 640,480);

		try{
			if(!uvcacquisition->returnQueue().empty()) {
				data = uvcacquisition->returnQueue().back();
			}
			else{
				//break;
			}

			resize(data, data, Size(640,480), 0, 0, INTER_NEAREST);
			minMaxLoc(data, &minVal, &maxVal, &minLoc, &maxLoc);
			img = raw_to_8bit(data);
			display_temperature(img, minVal, minLoc, Scalar(255, 0, 0));
			display_temperature(img, maxVal, maxLoc, Scalar(0, 0, 255));

			// Display frame
			if (!img.empty()) {
				imshow("Lepton Radiometry", img);
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
