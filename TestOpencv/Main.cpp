////#include <iostream>
////
//
////
////int main() {
////    cv::VideoCapture clip("D:\\opencv\\assets\\compressed.mp4");
////
////    int fps = 1000 / 25;
////
////    if (!clip.isOpened()) {
////        std::cerr << "Failed to open video file." << std::endl;
////        return -1;
////    }
////
////
////    cv::Mat firstFrame;
////    clip.read(firstFrame);
////
////    //cv::Mat first_frame;
////    //clip.read(first_frame);
////
////    int ogw = firstFrame.cols;
////    int ogh = firstFrame.rows;
////
////    std::cout << ogw << std::endl;
////
////    std::string window_name = "Fake Camera Controller";
////    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
////
////
////    VideoController controller(ogw, ogh);
////
////    cv::Mat finalFrame;
////
////    while (true) {
////
////        if (!clip.read(finalFrame)) {
////            break;
////        }
////
////        finalFrame = controller.ProcessFrame(finalFrame);
////
////        //controller.ProcessFrame(finalFrame);
////
////        int key = cv::waitKey(20);
////
////        if (key == '+' || key == '=') {
////            controller.cameraZoom *= 1.1;
////        } else if (key == '-') {
////            controller.cameraZoom /= 1.1;
////        } else if (key == 'w') {
////            controller.yShift -= static_cast<int>(10);
////        } else if (key == 's') {
////            controller.yShift += static_cast<int>(10);
////        } else if (key == 'a') {
////            controller.xShift -= static_cast<int>(10);
////        } else if (key == 'd') {
////            controller.xShift += static_cast<int>(10);
////        } else if (key == 27) {
////            break;
////        }
////        
////        
////        cv::imshow(window_name, finalFrame);
////    }
////
////    finalFrame.release();
////    clip.release();
////    cv::destroyAllWindows();
////
////    return 0;
////}
////
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

using namespace std;
using namespace TestOpencv;

//cv::Mat translateImg(cv::Mat & img, int offsetx, int offsety) {
//	cv::Mat trans_mat = (cv::Mat_<double>(2, 3) << 1, 0, offsetx, 0, 1, offsety);
//	warpAffine(img, img, trans_mat, img.size());	
//	trans_mat.release();
//	return img;
//}
//
//cv::cuda::GpuMat translateImg(cv::cuda::GpuMat& img, int offsetx, int offsety) {
//
//	//d_img.upload(img);
//
//	cv::Mat trans_mat = (cv::Mat_<double>(2, 3) << 1, 0, offsetx, 0, 1, offsety);
//	cv::cuda::warpAffine(img, img, trans_mat, img.size());
//
//	//d_result.download(img);
//	return img;
//}


void custom_hconcat(const cv::cuda::GpuMat src1, const cv::cuda::GpuMat src2, cv::cuda::GpuMat& result) {

	int size_cols = src1.cols + src2.cols;
	int size_rows = std::max(src1.rows, src2.rows);
	cv::cuda::GpuMat hconcat(size_rows, size_cols, src1.type());
	src2.copyTo(hconcat(cv::Rect(0, 0, src2.cols, src2.rows)));
	src1.copyTo(hconcat(cv::Rect(src2.cols, 0, src1.cols, src1.rows)));

	result = hconcat.clone();
}


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

	const std::string clipSaveName = "D:\\opencv\\assets\\final.avi";
	const std::string clipLeftName = "D:\\opencv\\assets\\Left_0009.mp4";
	const std::string clipRightName = "D:\\OPENCV\\Assets\\Right_0009.mp4";
	
	cv::VideoCapture clipLeft = cv::VideoCapture(clipLeftName);
	cv::VideoCapture clipRight = cv::VideoCapture(clipRightName);

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


	int finalResolutionW = 4000;
	int finalResolutionH = 3000;


	//cv::CAP_OPENCV_MJPEG
	cv::VideoWriter videoWriter = cv::VideoWriter(clipSaveName, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 25, cv::Size(finalResolutionW, finalResolutionH));
	//cv::Ptr<cv::cudacodec::VideoWriter> d_writer;
	//d_writer = cv::cudacodec::createVideoWriter(clipSaveName, cv::Size(finalResolutionW, finalResolutionH));
	//const String& fileName, const Size frameSize,

	cv::Ptr<cv::cudacodec::VideoReader> d_reader = cv::cudacodec::createVideoReader(clipLeftName);

	CORE_INFO("Original width: {0}, original height: {1}", originalWidth, originalHeight);

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
	

	while (true)
	{
		currentFrame++;
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

		//translateImg(frameRightGPU, 0, shift);

		//cv::Mat trans_mat = (cv::Mat_<double>(2, 3) << 1, 0, 0, 0, 1, 2);
		//warpaffina on gpu
		//cv::cuda::warpAffine(frameRightGPU, frameRightGPU, trans_mat, frameRightGPU.size());						

		//cv::cuda::GpuMat dst_gpu;
		cv::Size dsize = cv::Size(frameRightGPU.cols, frameRightGPU.rows + shift);

		// Apply the affine transformation using the GPU
		cv::cuda::warpAffine(frameRightGPU, frameRightGPU, M, dsize, cv::BORDER_CONSTANT);


		cv::cuda::remap(frameLeftGPU, frameLeftGPU, warpMatrixLeftMapXGPU,    warpMatrixLeftMapYGPU,  cv::INTER_LINEAR, cv::BORDER_CONSTANT);
		cv::cuda::remap(frameRightGPU, frameRightGPU, warpMatrixRightMapXGPU, warpMatrixRightMapYGPU, cv::INTER_LINEAR, cv::BORDER_CONSTANT);

		// Create the transformation matrix for the shift operation

		custom_hconcat(frameRightGPU, frameLeftGPU, finalFrameGPU);

		frameLeftGPU.release();
		frameRightGPU.release();


		cv::cuda::resize(finalFrameGPU, finalFrameGPU, cv::Size(finalResolutionW, finalResolutionH));

		finalFrameGPU.download(finalFrame);
		

		//cv::resize(finalFrame, finalFrame, cv::Size(1920, 1080));
		videoWriter.write(finalFrame);
		//d_writer->write(finalFrameGPU);
	/*	cv::imshow("finalFrame", finalFrame);
		cv::waitKey(10);*/
		CORE_INFO("Current frame: {0}", currentFrame);


		if (currentFrame >= 1000)
		{
			CORE_INFO("Finished");
			break;
		}
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
	videoWriter.release();

	cv::destroyAllWindows();

	return 0;			
}


//int main() {
//	
//	cv::VideoCapture clip("D:\\opencv\\assets\\final.avi");
//	clip.set(cv::CAP_PROP_FPS, 25);
//	//clip.set(cv::CAP_PROP_FRAME_WIDTH,4000);
//	//clip.set(cv::CAP_PROP_FRAME_HEIGHT,3000);
//	clip.set(cv::CAP_FFMPEG, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
//	std::cout << 'M' << 'J' << 'P' << 'G' << std::endl;
//
//	if (!clip.isOpened()) {
//		std::cerr << "Failed to open video file." << std::endl;
//		return -1;
//	}
//
//	cv::Mat frame; 
//
//	while (true) {
//		if (!clip.read(frame)) {
//			std::cout << "Finished" << std::endl;
//			break;
//		}
//		cv::resize(frame, frame, cv::Size(1920, 1080));
//		cv::imshow("Video", frame);
//		cv::waitKey(10);
//	}
//
//	frame.release();
//	clip.release();
//	cv::destroyAllWindows();
//}