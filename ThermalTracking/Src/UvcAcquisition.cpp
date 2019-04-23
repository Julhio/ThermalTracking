/*
 * UvcAcquisition.cpp
 *
 *  Created on: 23 de mar de 2019
 *      Author: Julhio Cesar Navas
 */

#include "../Inc/UvcAcquisition.h"

std::queue<cv::Mat> *frameQueue;

UvcAcquisition::UvcAcquisition()
{
	init();
}

UvcAcquisition::~UvcAcquisition() {
	// TODO Auto-generated destructor stub
}

std::queue<cv::Mat>* UvcAcquisition::returnQueue(){
	return frameQueue;
}

void UvcAcquisition::init() {
	uvc_error_t res;

	res = uvc_init(&ctx, NULL);	// Initialize a UVC service context.
	if (res < 0) {
		uvc_perror(res, "uvc_init");
		return; //res;
	}
	puts("UVC initialized");

	/* Locates the first attached UVC device, stores in dev */
	res = uvc_find_device(ctx, &dev, PT_USB_VID, PT_USB_PID, NULL);

	if (res < 0) {
		uvc_perror(res, "uvc_find_device"); /* no devices found */
		return;
	}

	puts("Device found");

	res = uvc_open(dev, &devh); /* Try to open the device: requires exclusive access */

	if (res < 0) {
		uvc_perror(res, "uvc_open"); /* unable to open device */

		/* Release the device descriptor */
		uvc_unref_device(dev);
		dev = NULL;
		return;
	}

	if (res < 0) {
		uvc_perror(res, "uvc_open"); /* unable to open device */
	}
	puts("Device opened");

	fmtVector *frame_formats = uvc_get_frame_formats_by_guid(devh, VS_FMT_GUID_Y16);

	if(sizeof(frame_formats) == 0){
		printf("device does not support Y16\n");
	} //else {

	setVideoFormat();
}

fmtVector *UvcAcquisition::uvc_get_frame_formats_by_guid(uvc_device_handle_t *devh, unsigned char *vs_fmt_guid){
	fmtVector *fmt = new fmtVector();
	uvc_frame_desc *p_frame_desc;
	const uvc_format_desc_t *p_format_desc;
	p_format_desc = uvc_get_format_descs(devh);
	const uvc_format_desc_t *format_desc;

	while(p_format_desc != NULL)
	{
		format_desc = p_format_desc;

		if(memcmp(vs_fmt_guid, format_desc->guidFormat, 4) == 0)
		{
			p_frame_desc = format_desc->frame_descs;

			while(p_frame_desc != 0)
			{
				fmt->push_back(p_frame_desc);
				p_frame_desc = p_frame_desc->next;
			}
			return fmt;
		}
		p_format_desc = p_format_desc->next;
	}
	return {};
}

void UvcAcquisition::frameCallback(uvc_frame_t *frame, void *userptr) {
	Mat Img_Source16Bit_Gray(frame->height, frame->width, CV_16UC1);

	Img_Source16Bit_Gray.data = reinterpret_cast<uchar*>(frame->data);

	if(frame->data_bytes != (2 * frame->width * frame->height))
		return;

	//Check if queue is full
	/*if(q.size() > )*/
	frameQueue->push(Img_Source16Bit_Gray);
}

void UvcAcquisition::setVideoFormat(){
	res = uvc_get_stream_ctrl_format_size(devh, &ctrl, UVC_FRAME_FORMAT_Y16, 80, 60, 9);

	/* Print out the result */
	//uvc_print_stream_ctrl(&ctrl, stderr);

	if (res < 0) {
		uvc_perror(res, "get_mode"); /* device doesn't provide a matching stream */
	}
}

void UvcAcquisition::pauseStream() {
    uvc_stop_streaming(devh);
    puts("Done streaming.");

	uvc_close(devh);		//Release our handle on the device
	puts("Device closed");

	uvc_unref_device(dev);	//Release the device descriptor

	uvc_exit(ctx);
	puts("UVC exited");
}

void UvcAcquisition::startStream() {
    uvc_start_streaming(devh, &ctrl, UvcAcquisition::frameCallback, &cb_ctx, 0);

    if (res < 0) {
        uvc_perror(res, "start_streaming"); /* unable to start stream */
        uvc_close(devh);
        puts("Device closed");

        return;
    }
    puts("Streaming...");
}




