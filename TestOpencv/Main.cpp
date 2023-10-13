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

//recording
#include "Recording/Soc_VideoWriter.h"
#include "Recording/Soc_VideoReader.h"

using namespace std;
using namespace TestOpencv;

//TODO: MOVE TO UTILS THIS IS ALSO IN VIDEOWRITER
void custom(const cv::cuda::GpuMat src1, const cv::cuda::GpuMat src2, cv::cuda::GpuMat& result)
{
	int size_cols = src1.cols + src2.cols;
	int size_rows = std::max(src1.rows, src2.rows);
	cv::cuda::GpuMat hconcat(size_rows, size_cols, src1.type());
	src1.copyTo(hconcat(cv::Rect(0, 0, src1.cols, src1.rows)));
	src2.copyTo(hconcat(cv::Rect(src1.cols, 0, src2.cols, src2.rows)));

	result = hconcat.clone();
}


int main() {
	Log::Init();

	CORE_INFO("Welcome to kaspoehh analyzing software version {}.{}.{}  !", 0, 0, 1);
	
	bool hasCuda = cv::cuda::getCudaEnabledDeviceCount() > 0;
	
	if (!hasCuda) {
		//std::cout << "CUDA is not available!" << std::endl;
		CORE_ERROR("CUDA is not available!");
		return -1;
	}
	CORE_INFO("CUDA is available!");

	bool saveVideo = false;
	bool playingWithGui = true;

	const cv::String clipSaveName = "D:\\opencv\\assets\\final.h264";
	
	
	const std::string clipSaveNameLocation = "D:\\opencv\\assets\\";
	const std::string clipLeftName = "D:\\opencv\\assets\\Left_0009.mp4";
	const std::string clipRightName = "D:\\OPENCV\\Assets\\Right_0009.mp4";

	Soc_VideoReader videoReader = Soc_VideoReader(clipLeftName, clipRightName);
	
	std::cout << cv::getBuildInformation() << std::endl;

	//check if the clip is open

	cv::cuda::GpuMat firstFrameLeft;
	cv::cuda::GpuMat firstFrameRight;

	videoReader.Read(firstFrameLeft, firstFrameRight);


	int originalWidth = firstFrameLeft.cols;
	int originalHeight = firstFrameRight.rows;

	int finalResolutionW = 1920;
	int finalResolutionH = 1080;

	SocVideoWriter videoWriter = SocVideoWriter(clipSaveName, 25.0, finalResolutionW, finalResolutionH);
	
	CORE_INFO("Original width: {0}, original height: {1}", originalWidth, originalHeight);
		
	//videoWriter.release();

	float cofLeft[] = { -0.199f, -0.1300, -0.0150 };
	float cofRight[] = { -0.4190, 0.0780, -0.0460 };

	cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 3000, 0, firstFrameLeft.cols / 2, 0, 3000, firstFrameLeft.rows / 2, 0, 0, 1);

	//release
	firstFrameLeft.release() ;
	firstFrameRight.release();

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

	int midShiftLeft = 310;
	int midShiftRightTopLeft = 260;
	int midShiftRightBottomLeft = 180;
	int shiftTotalHeightAdd = -700;


	//warp points in images up or down so they allign and its full screen
	cv::cuda::GpuMat warpMatrixLeftMapXGPU;
	cv::cuda::GpuMat warpMatrixLeftMapYGPU;

	cv::cuda::GpuMat warpMatrixRightMapXGPU;
	cv::cuda::GpuMat warpMatrixRightMapYGPU;

	{
		//left shift warp map
		{
			int offsetMidfield = 0;
			cv::Point2f srcPointsLeft[4];
			cv::Point2f dstPointsLeft[4];

			srcPointsLeft[0] = cv::Point2f(0, 0);
			srcPointsLeft[1] = cv::Point2f(0, originalHeight);
			srcPointsLeft[2] = cv::Point2f(originalWidth, 0);

			dstPointsLeft[0] = cv::Point2f(0, 0 + shiftTotalHeightAdd);									//top left
			dstPointsLeft[1] = cv::Point2f(0, originalHeight - offsetMidfield);											//bottom right
			dstPointsLeft[2] = cv::Point2f(originalWidth, midShiftLeft + shiftTotalHeightAdd);			//top right

			cv::Mat warpMatrixLeft;
			warpMatrixLeft = cv::getAffineTransform(srcPointsLeft, dstPointsLeft);

			cv::cuda::buildWarpAffineMaps(warpMatrixLeft, false, cv::Size(originalWidth, originalHeight), warpMatrixLeftMapXGPU, warpMatrixLeftMapYGPU);
			warpMatrixLeft.release();
		}


		//right shift warp map
		{
			cv::Point2f srcPoints[4];
			cv::Point2f dstPoints[4];

			srcPoints[0] = cv::Point2f(0, 0);
			srcPoints[1] = cv::Point2f(0, originalHeight);
			srcPoints[2] = cv::Point2f(originalWidth, 0);

			dstPoints[0] = cv::Point2f(0, midShiftRightTopLeft + shiftTotalHeightAdd); //top left
			dstPoints[1] = cv::Point2f(0, originalHeight + midShiftRightBottomLeft);   //bottom left
			dstPoints[2] = cv::Point2f(originalWidth, shiftTotalHeightAdd);		       //top right

			cv::Mat warpMatrixRight;
			warpMatrixRight = cv::getAffineTransform(srcPoints, dstPoints);

			cv::cuda::buildWarpAffineMaps(warpMatrixRight,
				false, cv::Size(originalWidth, originalHeight), warpMatrixRightMapXGPU, warpMatrixRightMapYGPU);
			warpMatrixRight.release();
		}
	}


	double avgTime = 0.0; // to calculate average time for all frames
	int frameCount = 0; // to keep track of the number of frames processed

	int custom1 = 0;
	int custom1_last = 0;
	int custom2 = 0;
	int custom2_last = 0;
	int custom3 = 0;
	int custom3_last = 0;
	int custom4 = 0;
	int custom4_last = 0;

	//KDB Opencv : Custom1: 100, custom2 : -400, custom3 : 210, custom4 : 540
	videoReader.ResetClips();

	int currentFrame = 0;

	cv::cuda::GpuMat mapRightFrameShiftX;
	cv::cuda::GpuMat mapRightFrameShiftY;

	//Calculate shift
	{
		int shift = 80;
		int ab = abs(shift);

		cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, 0, 0, 1, shift);

		cv::Size dsize = cv::Size(originalWidth, originalHeight + shift);

		cv::cuda::buildWarpAffineMaps(
			M, false, dsize, mapRightFrameShiftX, mapRightFrameShiftY
		);
	}


	cv::Mat finalFrameCPU;
	cv::cuda::GpuMat finalFrameGPU;

	//TODO: Get frames from the video reader
	int delay = 1000 / 25.0;

	while (true)
	{
		double startTicks = cv::getTickCount(); // start tick count
		clock_t startTime = clock();
		
		if (!videoReader.Read(frameLeftGPU, frameRightGPU))
		{
			break;
		}

		//remap so images are straight
		cv::cuda::remap(frameLeftGPU, frameLeftGPU, mapLeftXGPU, mapLeftYGPU, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
		cv::cuda::remap(frameRightGPU, frameRightGPU, mapRightXGPU, mapRightYGPU, cv::INTER_LINEAR, cv::BORDER_CONSTANT);


		//Shift the right frame on the y axis
		cv::cuda::remap(frameRightGPU, frameRightGPU, mapRightFrameShiftX, mapRightFrameShiftY, cv::INTER_LINEAR, cv::BORDER_CONSTANT);

		//Shift points in images up or down so they allign and its full screen
		cv::cuda::remap(frameLeftGPU, frameLeftGPU, warpMatrixLeftMapXGPU,    warpMatrixLeftMapYGPU,  cv::INTER_LINEAR, cv::BORDER_CONSTANT);
		cv::cuda::remap(frameRightGPU, frameRightGPU, warpMatrixRightMapXGPU, warpMatrixRightMapYGPU, cv::INTER_LINEAR, cv::BORDER_CONSTANT);

		if (saveVideo)
		{
			videoWriter.Write(frameLeftGPU, frameRightGPU, finalFrameGPU);
		}

		if (playingWithGui) {
			custom(frameLeftGPU, frameRightGPU, finalFrameGPU);
			cv::cuda::resize(finalFrameGPU, finalFrameGPU, cv::Size(finalResolutionW, finalResolutionH));
			finalFrameGPU.download(finalFrameCPU);
			cv::imshow("final", finalFrameCPU);
			
			int key = 0;
			while (clock() - startTime < delay) {
				key = cv::waitKey(1);
			}
			//q key
			if (key == 113)
			{
				break;
			}

			//convert ifs to switch
			switch (key)
			{
			case 119: //W
				custom1 += 10;
				break;
			case 115: //S
				custom1 -= 10;
				break;
			case 101: //E
				custom2 += 10;
				break;
			case 100: //d
				custom2 -= 10;
				break;
			case 114: //R
				custom3 += 10;
				break;
			case 102: //F
				custom3 -= 10;
				break;
			case 116: //T
				custom4 += 10;
				break;
			case 103: //G
				custom4 -= 10;
				break;
			}
		}


		double endTicks = cv::getTickCount(); // end tick count
		double timeInSeconds = (endTicks - startTicks) / cv::getTickFrequency();

	
		//std::cout << "Time taken for frame " << frameCount << ": " << timeInSeconds << " seconds" << std::endl;
		//CORE_INFO("Time taken for frame {0}: {1} seconds", frameCount, timeInSeconds);

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

	//videoWriter.release();

	cv::destroyAllWindows();

	return 0;			
}
