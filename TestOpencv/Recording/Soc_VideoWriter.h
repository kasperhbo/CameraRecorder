#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/shape/shape_transformer.hpp>
#include <opencv2/cudacodec.hpp>
#include <vector>
#include <chrono>
#include <iostream>
#include <cmath>

#include "../Utils/ImageUtil.h"

using namespace std;

class Soc_VideoWriter
{
public:
	Soc_VideoWriter(cv::String location, int width, int height, int fps);
	~Soc_VideoWriter();

	bool Initialize();

	void Write(cv::cuda::GpuMat& frameToShow);

private:
	int width;
	int height;

	cv::String location;
	int fps;

	cv::Ptr<cv::cudacodec::VideoWriter> d_writer;

	cv::cuda::Stream stream; 
};