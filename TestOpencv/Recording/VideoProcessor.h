#pragma once
#include <iostream>

#include "../Utils/Log.h"
#include "../Utils/TimeUtil.h"
#include "../Utils/CudaUtil.h"
#include "../Utils/ImageUtil.h"
#include "../Recording/Soc_VideoReader.h"
#include "../Recording/Soc_VideoWriter.h"
#include "../Recording/Soc_VideoViewer.h"


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
	

using namespace TestOpencv;


class VideoProcessor
{
public:
	VideoProcessor(
		const std::string fileLeft, const std::string fileRight, const std::string finalFile, 
		const bool saveVideo = false, const bool showVideo = true, const double fps = 25.0, const int width = 1920, const int height = 1080); 
	
	~VideoProcessor();

	bool Initialize();
	

	bool ProcessFrame();

	void ReleaseFrames();
	void ReleaseClips();
	void ReleaseGPUMats();
	void ReleaseCPUMats();
		
	bool Terminate();

private:
	void CreateImageMaps();
	void CreateBarrelUndistortMaps();
	void CreateShiftMap();
	void CreatePointShiftMaps();

private:

	const int rightFrameYShift = 80;

	const float cofLeft[3] = { -0.199f, -0.1300, -0.0150 };
	const float cofRight[3] = { -0.4190, 0.0780, -0.0460 };
		
	const std::string fileLeft;
	const std::string fileRight;
	const std::string finalFile;

	const bool saveVideo = true;
	const bool showVideo = true;

	const double fps;
	const int finalWidth;
	const int finalHeight;

	int originalWidth;
	int originalHeight;

	cv::cuda::GpuMat frameLeft  = cv::cuda::GpuMat();
	cv::cuda::GpuMat frameRight = cv::cuda::GpuMat();;
	cv::cuda::GpuMat frameFinal = cv::cuda::GpuMat();;

	//Barrel distort maps
	cv::cuda::GpuMat barrelUndistortMapLeftX;// = cv::cuda::GpuMat();;
	cv::cuda::GpuMat barrelUndistortMapLeftY;// = cv::cuda::GpuMat();;

	cv::cuda::GpuMat barrelUndistortMapRightX;//= cv::cuda::GpuMat();;
	cv::cuda::GpuMat barrelUndistortMapRightY;//= cv::cuda::GpuMat();;

	//Shift maps
	cv::cuda::GpuMat shiftMapRightX ;//= cv::cuda::GpuMat();;
	cv::cuda::GpuMat shiftMapRightY ;//= cv::cuda::GpuMat();;
	
	//Position change maps
	cv::cuda::GpuMat positionChangeMapLeftX ;//= cv::cuda::GpuMat();;
	cv::cuda::GpuMat positionChangeMapLeftY; //0 = cv::cuda::GpuMat();;
	
	cv::cuda::GpuMat positionChangeMapRightX; // = cv::cuda::GpuMat();;
	cv::cuda::GpuMat positionChangeMapRightY ;// = cv::cuda::GpuMat();;
	

	Soc_VideoReader * videoReader;
	Soc_VideoViewer * videoViewer;	
	Soc_VideoWriter * videoWriter;	
};
