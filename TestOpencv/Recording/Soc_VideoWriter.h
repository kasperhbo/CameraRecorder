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

#include <iostream>

#include "opencv2/opencv_modules.hpp"
	
#include <vector>
#include <numeric>

#include "opencv2/core.hpp"
#include "opencv2/cudacodec.hpp"
#include "opencv2/highgui.hpp"

using namespace std;

class Soc_VideoWriter
{
public:
	Soc_VideoWriter(const std::string location, double fps, int widthres, int heightres);
	~Soc_VideoWriter();

	void Write(const cv::cuda::GpuMat frameLeft, const cv::cuda::GpuMat frameRight, cv::cuda::GpuMat& result);

private:
	int widthres;
	int heightres;

	/*cv::VideoWriter writer;*/
	cv::Ptr<cv::cudacodec::VideoWriter> d_writer;

	cv::cuda::Stream stream; 
	
};