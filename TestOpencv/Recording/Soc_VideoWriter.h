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

#include "../Log.h"

using namespace std;
using namespace TestOpencv;

class SocVideoWriter
{
public:
	SocVideoWriter(const std::string location, double fps, int widthres, int heightres);
	~SocVideoWriter();

	void Write(const cv::cuda::GpuMat frameLeft, const cv::cuda::GpuMat frameRight, cv::cuda::GpuMat& result);

private:
	void CustomHConcat(const cv::cuda::GpuMat src1, const cv::cuda::GpuMat src2, cv::cuda::GpuMat& result);

private:
	int widthres;
	int heightres;

	/*cv::VideoWriter writer;*/
	cv::Ptr<cv::cudacodec::VideoWriter> d_writer;

	cv::cuda::Stream stream; 
	
};