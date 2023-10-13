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
#include "Log.h"

#include <fstream> // Include this at the top of your file

//make a timer
#include <chrono>

#include "Soc_VideoWriter.h"

using namespace std;
using namespace TestOpencv;



int main() {
	Log::Init();

	CORE_INFO("Welcome to vvoene analyzing software version {}.{}.{}  !", 0, 0, 1);
	
	bool hasCuda = cv::cuda::getCudaEnabledDeviceCount() > 0;
	
	if (!hasCuda) {
		//std::cout << "CUDA is not available!" << std::endl;
		CORE_ERROR("CUDA is not available!");
		return -1;
	}
	CORE_INFO("CUDA is available!");

	const cv::String clipSaveName = "D:\\opencv\\assets\\final.h264";
	
	
	const std::string clipSaveNameLocation = "D:\\opencv\\assets\\";
	const std::string clipLeftName = "D:\\opencv\\assets\\Left_0009.mp4";
	const std::string clipRightName = "D:\\OPENCV\\Assets\\Right_0009.mp4";
	
	cv::VideoCapture clipLeft = cv::VideoCapture(clipLeftName);
	cv::VideoCapture clipRight = cv::VideoCapture(clipRightName);
	
	std::cout << cv::getBuildInformation() << std::endl;

	//check if the clip is open

	if (!clipLeft.isOpened()) {
		CORE_ERROR("Failed to open video file {0}.", clipLeftName);
		return -1;
	}

	if (!clipRight.isOpened()) {
		CORE_ERROR("Failed to open video file {0}.", clipRightName);
		return -1;
	}

	cv::Mat firstFrame;

	clipLeft.read(firstFrame);

	int originalWidth = firstFrame.cols;
	int originalHeight = firstFrame.rows;


	int finalResolutionW = 8000;
	int finalResolutionH = 6000;


	//cv::CAP_OPENCV_MJPEG
	//cv::VideoWriter videoWriter = cv::VideoWriter(clipSaveName, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 25, cv::Size(finalResolutionW, finalResolutionH));

	SocVideoWriter videoWriter = SocVideoWriter(clipSaveName, 25.0, 4000, 3000);
	
	CORE_INFO("Original width: {0}, original height: {1}", originalWidth, originalHeight);
		
	//videoWriter.release();

	float cofLeft[] = { -0.199f, -0.1300, -0.0150 };
	float cofRight[] = { -0.4190, 0.0780, -0.0460 };

	cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 3000, 0, firstFrame.cols / 2, 0, 3000, firstFrame.rows / 2, 0, 0, 1);

	//create the left maps
	cv::Mat mapLeftX, mapLeftY;		
	cv::Mat distCoeffsLeft = (cv::Mat_<double>(1, 5) << cofLeft[0], cofLeft[1], cofLeft[2], 0, 0);

	cv::initUndistortRectifyMap(
		cameraMatrix,
		distCoeffsLeft,
		cv::Mat(),
		cameraMatrix,
		cv::Size(4000, 4000),
		CV_32FC1,
		mapLeftX,
		mapLeftY
	);


	//Create the right maps
	cv::Mat mapRightX, mapRightY;
	cv::Mat distCoeffsRight = (cv::Mat_<double>(1, 5) << cofRight[0], cofRight[1], cofRight[2], 0, 0);

	cv::initUndistortRectifyMap(
		cameraMatrix,
		distCoeffsRight,
		cv::Mat(),
		cameraMatrix,
		cv::Size(4000, 4000),
		CV_32FC1,
		mapRightX,
		mapRightY
	);

	//release camera matrix
	cameraMatrix.release();

	//create the frames mats for the clips
	cv::Mat frameLeft;
	cv::Mat frameRight;
	cv::Mat finalFrame;

	cv::Mat warpMatrixLeft;
	cv::Mat warpMatrixRight;

	//create the gpu mats for the frames
	cv::cuda::GpuMat frameLeftGPU;
	cv::cuda::GpuMat frameRightGPU;

	cv::cuda::GpuMat mapLeftXGPU;
	cv::cuda::GpuMat mapLeftYGPU;
	mapLeftXGPU.upload(mapLeftX);
	mapLeftYGPU.upload(mapLeftY);

	cv::cuda::GpuMat mapRightXGPU;
	cv::cuda::GpuMat mapRightYGPU;
	mapRightXGPU.upload(mapRightX);
	mapRightYGPU.upload(mapRightY);

	int shift = 80;
	int midShiftLeft = 310;
	int midShiftRightTopLeft = 260;
	int midShiftRightBottomLeft = 180;
	int ab = abs(shift);


	cv::Point2f srcPoints[4];
	cv::Point2f dstPoints[4];

	srcPoints[0] = cv::Point2f(0, 0);
	srcPoints[1] = cv::Point2f(0, originalHeight);
	srcPoints[2] = cv::Point2f(originalWidth, 0);
	srcPoints[3] = cv::Point2f(originalWidth, originalHeight);

	int currentFrame = 0;

	//left
	int shiftTotalHeightAdd = -700;
	dstPoints[0] = cv::Point2f(0, 0 + shiftTotalHeightAdd);									//top left
	dstPoints[1] = cv::Point2f(0, originalHeight);											//bottom left
	dstPoints[2] = cv::Point2f(originalWidth, midShiftLeft + shiftTotalHeightAdd);			//top right
	dstPoints[3] = cv::Point2f(originalWidth, originalHeight);								//bottom right
	warpMatrixLeft = cv::getAffineTransform(srcPoints, dstPoints);
	//cv::warpAffine(frameLeft, frameLeft, warpMatrixLeft, frameLeft.size());
	//cv::cuda::buildWarpAffineMaps(warpMatrixLeft, false, cv::Size(originalWidth, originalHeight), mapLeftXGPU, mapLeftYGPU);
	cv::cuda::GpuMat warpMatrixLeftMapXGPU;
	cv::cuda::GpuMat warpMatrixLeftMapYGPU;
	
	cv::cuda::buildWarpAffineMaps(warpMatrixLeft, false, cv::Size(originalWidth, originalHeight), warpMatrixLeftMapXGPU, warpMatrixLeftMapYGPU);
	warpMatrixLeft.release();

	//right
	dstPoints[0] = cv::Point2f(0, midShiftRightTopLeft + shiftTotalHeightAdd); //top left
	dstPoints[1] = cv::Point2f(0, originalHeight + midShiftRightBottomLeft);   //bottom left
	dstPoints[2] = cv::Point2f(originalWidth, shiftTotalHeightAdd);		       //top right
	dstPoints[3] = cv::Point2f(originalWidth, originalHeight);				   //bottom right
	warpMatrixRight = cv::getAffineTransform(srcPoints, dstPoints);

	cv::cuda::GpuMat warpMatrixRightMapXGPU;
	cv::cuda::GpuMat warpMatrixRightMapYGPU;

	cv::cuda::buildWarpAffineMaps(warpMatrixRight, false, cv::Size(originalWidth, originalHeight), warpMatrixRightMapXGPU, warpMatrixRightMapYGPU);
	warpMatrixRight.release();



	cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, 0, 0, 1, shift);
	cv::cuda::GpuMat finalFrameGPU;

	int length = clipLeft.get(cv::CAP_PROP_FRAME_COUNT);

	double avgTime = 0.0; // to calculate average time for all frames
	int frameCount = 0; // to keep track of the number of frames processed

	while (true)
	{
		double startTicks = cv::getTickCount(); // start tick count
		
		//one of the clips is finished
		if (!clipLeft.read(frameLeft))
		{
			CORE_INFO("Left clip is finished");

			break;
		}
		if (!clipRight.read(frameRight))
		{
			CORE_INFO("Right clip is finished");

			break;
		}

		//remap on the gpu
		frameLeftGPU.upload(frameLeft);
		frameRightGPU.upload(frameRight);

		cv::cuda::remap(frameLeftGPU, frameLeftGPU, mapLeftXGPU, mapLeftYGPU, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
		cv::cuda::remap(frameRightGPU, frameRightGPU, mapRightXGPU, mapRightYGPU, cv::INTER_LINEAR, cv::BORDER_CONSTANT);

		//cv::cuda::GpuMat dst_gpu;
		cv::Size dsize = cv::Size(frameRightGPU.cols, frameRightGPU.rows + shift);

		// Apply the affine transformation using the GPU
		cv::cuda::warpAffine(frameRightGPU, frameRightGPU, M, dsize, cv::BORDER_CONSTANT);


		cv::cuda::remap(frameLeftGPU, frameLeftGPU, warpMatrixLeftMapXGPU,    warpMatrixLeftMapYGPU,  cv::INTER_LINEAR, cv::BORDER_CONSTANT);
		cv::cuda::remap(frameRightGPU, frameRightGPU, warpMatrixRightMapXGPU, warpMatrixRightMapYGPU, cv::INTER_LINEAR, cv::BORDER_CONSTANT);

		videoWriter.Write(frameLeftGPU, frameRightGPU, finalFrameGPU);


		frameLeftGPU.release();
		frameRightGPU.release();


		double endTicks = cv::getTickCount(); // end tick count
		double timeInSeconds = (endTicks - startTicks) / cv::getTickFrequency();

	
		//std::cout << "Time taken for frame " << frameCount << ": " << timeInSeconds << " seconds" << std::endl;
		CORE_INFO("Time taken for frame {0}: {1} seconds", frameCount, timeInSeconds);

		avgTime += timeInSeconds;
		frameCount++;
	}


	avgTime /= frameCount;
	std::cout << "Average time taken per frame: " << avgTime << " seconds" << std::endl;
	// Get the current time point
	auto now = std::chrono::system_clock::now();

	// Convert it to a time_t object
	std::time_t currentTime = std::chrono::system_clock::to_time_t(now);

	// Use a safer method to get localtime based on the platform
	std::tm localTime;

#if defined(_WIN32) || defined(_WIN64)  // Windows-specific code
	localtime_s(&localTime, &currentTime);
#else  // POSIX-specific code
	localtime_r(&currentTime, &localTime);
#endif

	// Use a stringstream to format the time into a string suitable for filenames
	std::stringstream ss;
	ss << localTime.tm_year + 1900 << "-"
		<< localTime.tm_mon + 1 << "-"
		<< localTime.tm_mday << "_"
		<< localTime.tm_hour << "-"
		<< localTime.tm_min << "-"
		<< localTime.tm_sec << "_data.txt";

	// Use the formatted time string as the filename
	std::string filename = ss.str();

	// Open an output file stream with the constructed filename
	std::ofstream outFile(clipSaveNameLocation  + filename);

	if (outFile.is_open()) {
		outFile << "Resolution: " << finalResolutionW << "x" << finalResolutionH << std::endl;
		outFile << "Average time taken per frame: " << avgTime << " seconds" << std::endl;
			
		outFile.close();
	}
	else {
		std::cerr << "Unable to open file for writing." << std::endl;
	}

	//d_reader.release();

	//release all the memory
	frameLeft.release();
	frameRight.release();
	finalFrame.release();

	frameLeftGPU.release();
	frameRightGPU.release();
	finalFrameGPU.release();

	mapLeftX.release();
	mapLeftY.release();
	mapRightX.release();
	mapRightY.release();

	mapLeftXGPU.release();
	mapLeftYGPU.release();
	mapRightXGPU.release();
	mapRightYGPU.release();

	warpMatrixLeftMapXGPU.release();
	warpMatrixLeftMapYGPU.release();
	warpMatrixRightMapXGPU.release();
	warpMatrixRightMapYGPU.release();

	clipLeft.release();
	clipRight.release();
	//videoWriter.release();

	cv::destroyAllWindows();

	return 0;			
}
