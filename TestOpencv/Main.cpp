#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/shape/shape_transformer.hpp>
#include <opencv2/cudacodec.hpp>

//recording
#include "Recording/Soc_VideoWriter.h"
#include "Recording/Soc_VideoReader.h"
#include "Recording/VideoProcessor.h"

#include "Utils/CudaUtil.h"
#include "Utils/TimeUtil.h"
#include "Utils/Log.h"

using namespace std;
using namespace TestOpencv;

int main() {
	bool saveVideo = true;
	bool playingWithGui = true;

	const std::string clipSaveNameLocation = "D:\\opencv\\assets\\saves\\";
	const cv::String clipSaveName = clipSaveNameLocation + TimeUtil::GetFormattedString() + "final.h264";
	const std::string clipLeftName = "D:\\opencv\\assets\\Recordings\\Left_0009.mp4";
	const std::string clipRightName = "D:\\OPENCV\\Assets\\Recordings\\Right_0009.mp4";

	//Soc_VideoReader videoReader = Soc_VideoReader(clipLeftName, clipRightName);
	//const std::string fileLeft, const std::string fileRight, const std::string finalFile, const bool saveVideo, const bool showVideo, const double fps, const int width, const int height
	Log::Init();
	VideoProcessor* processor = new VideoProcessor(clipLeftName, clipRightName, clipSaveName, false, true, 25.0, 1920, 1080);
	processor->Initialize();

	float delay = 1000 / 25.0;

	while (true) {
		clock_t startTime = clock();

		if (!processor->ProcessFrame())
		{
			break;
		}

		while (clock() - startTime < delay) {
			int key = cv::waitKey(1);
		}

	}

	return 0;
}
//	//videoWriter.release();
//
//	float cofLeft[] = { -0.199f, -0.1300, -0.0150 };
//	float cofRight[] = { -0.4190, 0.0780, -0.0460 };
//
//	cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 3000, 0, firstFrameLeft.cols / 2, 0, 3000, firstFrameLeft.rows / 2, 0, 0, 1);
//
//	//release
//	firstFrameLeft.release() ;
//	firstFrameRight.release();
//
//	//create the left maps
//	cv::Mat mapLeftX, mapLeftY;		
//	cv::Mat distCoeffsLeft = (cv::Mat_<double>(1, 5) << cofLeft[0], cofLeft[1], cofLeft[2], 0, 0);
//
//	cv::initUndistortRectifyMap(
//		cameraMatrix,
//		distCoeffsLeft,
//		cv::Mat(),
//		cameraMatrix,
//		cv::Size(4000, 4000),
//		CV_32FC1,
//		mapLeftX,
//		mapLeftY
//	);
//
//
//	//Create the right maps
//	cv::Mat mapRightX, mapRightY;
//	cv::Mat distCoeffsRight = (cv::Mat_<double>(1, 5) << cofRight[0], cofRight[1], cofRight[2], 0, 0);
//
//	cv::initUndistortRectifyMap(
//		cameraMatrix,
//		distCoeffsRight,
//		cv::Mat(),
//		cameraMatrix,
//		cv::Size(4000, 4000),
//		CV_32FC1,
//		mapRightX,
//		mapRightY
//	);
//
//	//release camera matrix
//	cameraMatrix.release();
//
//	//create the gpu mats for the frames
//	cv::cuda::GpuMat frameLeftGPU;
//	cv::cuda::GpuMat frameRightGPU;
//
//	cv::cuda::GpuMat mapLeftXGPU;
//	cv::cuda::GpuMat mapLeftYGPU;
//
//	mapLeftXGPU.upload(mapLeftX);
//	mapLeftYGPU.upload(mapLeftY);
//
//	cv::cuda::GpuMat mapRightXGPU;
//	cv::cuda::GpuMat mapRightYGPU;
//
//	mapRightXGPU.upload(mapRightX);
//	mapRightYGPU.upload(mapRightY);
//
//	int midShiftLeft = 310;
//	int midShiftRightTopLeft = 260;
//	int midShiftRightBottomLeft = 180;
//	int shiftTotalHeightAdd = -700;
//
//
//	//warp points in images up or down so they allign and its full screen
//	cv::cuda::GpuMat warpMatrixLeftMapXGPU;
//	cv::cuda::GpuMat warpMatrixLeftMapYGPU;
//
//	cv::cuda::GpuMat warpMatrixRightMapXGPU;
//	cv::cuda::GpuMat warpMatrixRightMapYGPU;
//
//	{
//		//left shift warp map
//		{
//			int offsetMidfield = 0;
//			cv::Point2f srcPointsLeft[4];
//			cv::Point2f dstPointsLeft[4];
//
//			srcPointsLeft[0] = cv::Point2f(0, 0);
//			srcPointsLeft[1] = cv::Point2f(0, originalHeight);
//			srcPointsLeft[2] = cv::Point2f(originalWidth, 0);
//
//			dstPointsLeft[0] = cv::Point2f(0, 0 + shiftTotalHeightAdd);									//top left
//			dstPointsLeft[1] = cv::Point2f(0, originalHeight - offsetMidfield);											//bottom right
//			dstPointsLeft[2] = cv::Point2f(originalWidth, midShiftLeft + shiftTotalHeightAdd);			//top right
//
//			cv::Mat warpMatrixLeft;
//			warpMatrixLeft = cv::getAffineTransform(srcPointsLeft, dstPointsLeft);
//
//			cv::cuda::buildWarpAffineMaps(warpMatrixLeft, false, cv::Size(originalWidth, originalHeight), warpMatrixLeftMapXGPU, warpMatrixLeftMapYGPU);
//			warpMatrixLeft.release();
//		}
//
//
//		//right shift warp map
//		{
//			cv::Point2f srcPoints[4];
//			cv::Point2f dstPoints[4];
//
//			srcPoints[0] = cv::Point2f(0, 0);
//			srcPoints[1] = cv::Point2f(0, originalHeight);
//			srcPoints[2] = cv::Point2f(originalWidth, 0);
//
//			dstPoints[0] = cv::Point2f(0, midShiftRightTopLeft + shiftTotalHeightAdd); //top left
//			dstPoints[1] = cv::Point2f(0, originalHeight + midShiftRightBottomLeft);   //bottom left
//			dstPoints[2] = cv::Point2f(originalWidth, shiftTotalHeightAdd);		       //top right
//
//			cv::Mat warpMatrixRight;
//			warpMatrixRight = cv::getAffineTransform(srcPoints, dstPoints);
//
//			cv::cuda::buildWarpAffineMaps(warpMatrixRight,
//				false, cv::Size(originalWidth, originalHeight), warpMatrixRightMapXGPU, warpMatrixRightMapYGPU);
//			warpMatrixRight.release();
//		}
//	}
//
//
//	double avgTime = 0.0; // to calculate average time for all frames
//	int frameCount = 0; // to keep track of the number of frames processed
//
//	//KDB Opencv : Custom1: 100, custom2 : -400, custom3 : 210, custom4 : 540
//	videoReader.ResetClips();
//
//	int currentFrame = 0;
//
//	cv::cuda::GpuMat mapRightFrameShiftX;
//	cv::cuda::GpuMat mapRightFrameShiftY;
//
//	//Calculate shift
//	{
//		int shift = 80;
//		int ab = abs(shift);
//
//		cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, 0, 0, 1, shift);
//
//		cv::Size dsize = cv::Size(originalWidth, originalHeight + shift);
//
//		cv::cuda::buildWarpAffineMaps(
//			M, false, dsize, mapRightFrameShiftX, mapRightFrameShiftY
//		);
//	}
//
//
//	cv::Mat finalFrameCPU;
//	cv::cuda::GpuMat finalFrameGPU;
//
//	//TODO: Get frames from the video reader
//	int delay = 1000 / 25.0;
//
//	while (true)
//	{
//		double startTicks = cv::getTickCount(); // start tick count
//		clock_t startTime = clock();
//
//		CORE_INFO("Frame: {0}", frameCount);
//		
//		if (!videoReader.Read(frameLeftGPU, frameRightGPU))
//		{
//			break;
//		}
//
//		//remap so images are straight
//		cv::cuda::remap(frameLeftGPU, frameLeftGPU, mapLeftXGPU, mapLeftYGPU, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
//		cv::cuda::remap(frameRightGPU, frameRightGPU, mapRightXGPU, mapRightYGPU, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
//
//
//		//Shift the right frame on the y axis
//		cv::cuda::remap(frameRightGPU, frameRightGPU, mapRightFrameShiftX, mapRightFrameShiftY, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
//
//		//Shift points in images up or down so they allign and its full screen
//		cv::cuda::remap(frameLeftGPU, frameLeftGPU, warpMatrixLeftMapXGPU,    warpMatrixLeftMapYGPU,  cv::INTER_LINEAR, cv::BORDER_CONSTANT);
//		cv::cuda::remap(frameRightGPU, frameRightGPU, warpMatrixRightMapXGPU, warpMatrixRightMapYGPU, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
//
//		if (saveVideo)
//		{
//			videoWriter.Write(frameLeftGPU, frameRightGPU, finalFrameGPU);
//		}
//
//		if (playingWithGui) {
//			CudaUtil::CustomHConcat(frameLeftGPU, frameRightGPU, finalFrameGPU);
//			cv::cuda::resize(finalFrameGPU, finalFrameGPU, cv::Size(finalResolutionW, finalResolutionH));
//			finalFrameGPU.download(finalFrameCPU);
//			cv::imshow("final", finalFrameCPU);
//			
//			int key = 0;
//			while (clock() - startTime < delay) {
//				key = cv::waitKey(1);
//			}
//			//q key
//			if (key == 113)
//			{
//				break;
//			}
//		}
//
//
//		double endTicks = cv::getTickCount(); // end tick count
//		double timeInSeconds = (endTicks - startTicks) / cv::getTickFrequency();
//
//	
//		//std::cout << "Time taken for frame " << frameCount << ": " << timeInSeconds << " seconds" << std::endl;
//		//CORE_INFO("Time taken for frame {0}: {1} seconds", frameCount, timeInSeconds);
//
//		if(frameCount > 100)
//		{
//			break;
//		}
//
//		avgTime += timeInSeconds;
//		frameCount++;
//	}
//
//
//	avgTime /= frameCount;
//	std::cout << "Average time taken per frame: " << avgTime << " seconds" << std::endl;
//	// Get the current time point
//	{
//		// Open an output file stream with the constructed filename
//		std::ofstream outFile(clipSaveNameLocation + TimeUtil::GetFormattedString() + "log.txt");
//
//		if (outFile.is_open()) {
//			outFile << "Resolution: " << finalResolutionW << "x" << finalResolutionH << std::endl;
//			outFile << "Average time taken per frame: " << avgTime << " seconds" << std::endl;
//			outFile.close();
//		}
//		else {
//			std::cerr << "Unable to open file for writing." << std::endl;
//		}
//	}
//
//
//
//	frameLeftGPU.release();
//	frameRightGPU.release();
//	finalFrameGPU.release();
//
//	mapLeftX.release();
//	mapLeftY.release();
//	mapRightX.release();
//	mapRightY.release();
//
//	mapLeftXGPU.release();
//	mapLeftYGPU.release();
//
//	mapRightXGPU.release();
//	mapRightYGPU.release();
//
//	warpMatrixLeftMapXGPU.release();
//	warpMatrixLeftMapYGPU.release();
//
//	warpMatrixRightMapXGPU.release();
//	warpMatrixRightMapYGPU.release();
//
//	//videoWriter.release();
//
//	cv::destroyAllWindows();
//
//	return 0;			
//}
