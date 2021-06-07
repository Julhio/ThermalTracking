/*
 * UvcAcquisition.h
 *
 *  Created on: 23 de mar de 2019
 *      Author: Julhio Cesar Navas
 */

#ifndef INC_UvcAcquisition_H_
#define INC_UvcAcquisition_H_

#include <stdio.h>
#include <vector>
#include <libuvc/libuvc.h>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include <queue>

#define PT_USB_VID				0x1E4E
#define PT_USB_PID				0x0100

using namespace cv;
using namespace std;

typedef std::vector<uvc_frame_desc*> fmtVector;

class UvcAcquisition {
public:
	UvcAcquisition();
	uvc_error_t init();
	fmtVector *uvc_get_frame_formats_by_guid(uvc_device_handle_t *devh, unsigned char *vs_fmt_guid);
	void setVideoFormat();
	static void frameCallback(uvc_frame_t *frame, void *userptr);
	void startStream();
	void pauseStream();
	std::queue<cv::Mat> returnQueue();
	virtual ~UvcAcquisition();
	bool isConnected();

	struct cb_context {
		FILE *out;
		struct timeval tv_start;
		int frames;
	};

protected:
	unsigned char VS_FMT_GUID_Y16[16] = {'Y','1','6',' ', 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9b, 0x71};
	struct cb_context cb_ctx = {0};
	uvc_context_t *ctx;
	uvc_device_t *dev;
	uvc_device_handle_t *devh;
	uvc_stream_ctrl_t ctrl;
	uvc_error_t res;
	Mat img;
	bool uvc_connected;
private:
};

#endif /* INC_UvcAcquisition_H_ */
