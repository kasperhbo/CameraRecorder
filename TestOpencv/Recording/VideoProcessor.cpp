#include "VideoProcessor.h"

/// <summary>
/// Constructor
/// </summary>
/// <param name="fileLeft">Left video file path</param>
/// <param name="fileRight"> Right video file path</param>
/// <param name="finalFile"> Final video file path</param>
/// <param name="saveVideo"> Save the video</param>
/// <param name="showVideo"> Show the video</param>
/// <param name="fps">       The fps</param>
/// <param name="width">     The target width</param>
/// <param name="height">    The target height</param>
VideoProcessor::VideoProcessor(const std::string fileLeft, const std::string fileRight, const std::string finalFile, const bool saveVideo, const bool showVideo, const double fps, const int width, const int height)
	: fileLeft(fileLeft), fileRight(fileRight), finalFile(finalFile), saveVideo(saveVideo), showVideo(showVideo), fps(fps), finalWidth(width), finalHeight(height)
{

}

/// <summary>
/// Destructor
/// </summary>
VideoProcessor::~VideoProcessor()
{
	Terminate();
}

/// <summary>
/// Initialize the video processor
/// </summary>
/// <returns> If the video processor was initialized succesfully</returns>
bool VideoProcessor::Initialize()
{

	CORE_INFO(
		"VideoProcessor: Initializing. FileLeft: {0}, FileRight: {1}, FinalFile: {2}, SaveVideo: {3}, ShowVideo: {4}, FPS: {5}, Width: {6}, Height: {7}",
		fileLeft, fileRight, finalFile, saveVideo, showVideo, fps, finalWidth, finalHeight);

	videoReader = new Soc_VideoReader(fileLeft, fileRight);

	if (videoReader->Initialize())
	{
		CORE_INFO("Soc_VideoReader: Initialized {} ");

		cv::cuda::GpuMat* firstFrameLeft = new cv::cuda::GpuMat();
		cv::cuda::GpuMat* firstFrameRight = new cv::cuda::GpuMat();

		videoReader->Read(*firstFrameLeft, *firstFrameRight);

		originalWidth = firstFrameLeft->cols;
		originalHeight = firstFrameRight->rows;

		firstFrameLeft->release();
		firstFrameRight->release();

		CORE_INFO("Original width: {0}, original height: {1}", originalWidth, originalHeight);
	}
	else
	{
		CORE_CRITICAL("Soc_VideoReader: Failed to initialize");
		return false;
	}

	if (saveVideo)
	{
		videoWriter = new Soc_VideoWriter(finalFile, finalWidth, finalHeight, fps);

		if (videoWriter->Initialize())
		{
			CORE_INFO("Soc_VideoWriter: Initialized");
		}
		else
		{
			CORE_CRITICAL("Soc_VideoWriter: Failed to initialize");
			return false;
		}
	}
	if (showVideo)
	{
		videoViewer = new Soc_VideoViewer(finalWidth, finalHeight);
		if (videoViewer->Initialize())
		{
			CORE_INFO("Soc_VideoViewer: Initialized");
		}
		else
		{
			CORE_CRITICAL("Soc_VideoViewer: Failed to initialize");
			return false;
		}
	}

	CreateImageMaps();


	//create the left maps

	CORE_INFO("VideoProcessor: Initialized");
	return true;
}

/// <summary>
/// Process the frame
/// </summary>
/// <returns>If the frame was processed succesfully</returns>
bool VideoProcessor::ProcessFrame()
{
	if (videoReader->Read(frameLeft, frameRight))
	{
		CORE_INFO(
			"VideoProcessor: Processig.... FileLeft: {0}, FileRight: {1}, FinalFile: {2}, SaveVideo: {3}, ShowVideo: {4}, FPS: {5}, Width: {6}, Height: {7}",
			fileLeft, fileRight, finalFile, saveVideo, showVideo, fps, finalWidth, finalHeight);

		//Undistort frames
		CudaUtil::Remap(frameLeft, frameLeft, barrelUndistortMapLeftX, barrelUndistortMapLeftY);
		CudaUtil::Remap(frameRight, frameRight, barrelUndistortMapRightX, barrelUndistortMapRightY);

		//Shift right frame
		CudaUtil::Remap(frameRight, frameRight, shiftMapRightX, shiftMapRightY);

		//Shift frame points
		CudaUtil::Remap(frameLeft, frameLeft, positionChangeMapLeftX, positionChangeMapLeftY);
		CudaUtil::Remap(frameRight, frameRight, positionChangeMapRightX, positionChangeMapRightY);

		//Concat frames
		ImageUtil::HConcat(frameLeft, frameRight, frameFinal);

		//Resize the image to the output width and height
		ImageUtil::ResizeImage(frameFinal, frameFinal, finalWidth, finalHeight);

		//Write the frame
	/*	if (saveVideo)
		{
			videoWriter->Write(frameFinal);
		}*/

		if (showVideo)
		{
			videoViewer->ShowVideo(frameFinal);
		}

		return true;
	}
	else
	{
		CORE_INFO("VideoProcessor: Finished processing");
		return false;
	}

}

/// <summary>
/// Cleanup all the frames
/// </summary>
void VideoProcessor::ReleaseFrames()
{
}


/// <summary>
/// Cleanup all the clips
/// </summary>
void VideoProcessor::ReleaseClips()
{

}

/// <summary>
/// Cleanup all the GPU mats
/// </summary>
void VideoProcessor::ReleaseGPUMats() {

}

/// <summary>
/// Cleanup all cpu mats
/// </summary>
void VideoProcessor::ReleaseCPUMats() {

}


/// <summary>
/// Cleanup
/// </summary>
/// <returns></returns>
bool VideoProcessor::Terminate()
{
	void ReleaseFrames();
	void ReleaseClips();
	void ReleaseGPUMats();
	void ReleaseCPUMats();
}

/// <summary>
/// Create the image maps
/// </summary>
void VideoProcessor::CreateImageMaps() {
	CreateBarrelUndistortMaps();
	CreateShiftMap();
	CreatePointShiftMaps();
}

/// <summary>
/// Remove Barrel distortion
/// </summary>
void VideoProcessor::CreateBarrelUndistortMaps()
{
	cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 3000, 0, 4000 / 2, 0, 3000, 3000 / 2, 0, 0, 1);
	//barrel undistort
	CudaUtil::InitializeBarrelGPUMap(cofLeft, cameraMatrix, barrelUndistortMapLeftX, barrelUndistortMapLeftY);
	CudaUtil::InitializeBarrelGPUMap(cofRight, cameraMatrix, barrelUndistortMapRightX, barrelUndistortMapRightY);
	
}

/// <summary>
/// Shift y map up
/// </summary>
void VideoProcessor::CreateShiftMap() {
	//right
	CORE_INFO("VideoProcessor: Creating shift map");
	CudaUtil::MakeShiftYMap(rightFrameYShift, originalWidth, originalHeight, shiftMapRightX, shiftMapRightY);
}

/// <summary>
/// Stretch the image top so we dont have black bars
/// </summary>
void VideoProcessor::CreatePointShiftMaps() {
	//Position change
	int midShiftLeft = 310;
	int midShiftRightTopLeft = 260;
	int midShiftRightBottomLeft = 180;
	int shiftTotalHeightAdd = -700;

	//Left
	int offsetMidfield = 0;
	cv::Point2f srcPointsLeft[3] = {
		cv::Point2f(0, 0),//top left
		cv::Point2f(0, originalHeight),//bottom left
		cv::Point2f(originalWidth, 0),//top right
	};

	cv::Point2f dstPointsLeft[3] = {
		cv::Point2f(0, 0 + shiftTotalHeightAdd)        , //top left
		cv::Point2f(0, originalHeight - offsetMidfield),//bottom right
		cv::Point2f(originalWidth, midShiftLeft + shiftTotalHeightAdd)//top right
	};

	CudaUtil::MakeWarpAffineMaps(srcPointsLeft, dstPointsLeft, positionChangeMapLeftX, positionChangeMapLeftY, originalWidth, originalHeight);

	//Right
	cv::Point2f srcPointsRight[3] = {
		cv::Point2f(0, 0),//top left
		cv::Point2f(0, originalHeight),//bottom left
		cv::Point2f(originalWidth, 0),//top right
	};

	cv::Point2f dstPointsRight[3] = {
		cv::Point2f(0, midShiftRightTopLeft + shiftTotalHeightAdd),//top left
		cv::Point2f(0, originalHeight + midShiftRightBottomLeft),//bottom left
		cv::Point2f(originalWidth, shiftTotalHeightAdd)//top right
	};

	CudaUtil::MakeWarpAffineMaps(srcPointsRight, dstPointsRight, positionChangeMapRightX, positionChangeMapRightY, originalWidth, originalHeight);

}