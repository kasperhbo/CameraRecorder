#pragma once
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

#include "../Utils/ImageUtil.h"
#include "../Utils/Log.h"


using namespace TestOpencv;

class Soc_VideoViewer
{
public:
	Soc_VideoViewer(const int width, const int height, const std::string title = "VideoFrame");
	~Soc_VideoViewer();

	bool Initialize();

	void ShowVideo(cv::Mat& frame);
	void ShowVideo(cv::cuda::GpuMat& frame);

private:
	const int width;
	const int height;
	const std::string title;
};

