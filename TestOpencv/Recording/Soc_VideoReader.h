#pragma once

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

class Soc_VideoReader
{
	public:
		Soc_VideoReader(const cv::String locationLeft, const cv::String locationRight);
		~Soc_VideoReader();

		bool Initialize();

		void ResetClips();

		bool Read(cv::cuda::GpuMat& resultLeft, cv::cuda::GpuMat& resultRight);

		double GetFPS() { return fps; }

	private:
		cv::String locationLeft;
		cv::String locationRight;

		double fps;
		int videoLength;
		

		cv::Ptr<cv::cudacodec::VideoReader> d_readerLeft ;
		cv::Ptr<cv::cudacodec::VideoReader> d_readerRight;

		//cv::VideoCapture clipLeft;
		//cv::VideoCapture clipRight;

};

