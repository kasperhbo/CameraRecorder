#include "Soc_VideoViewer.h"

Soc_VideoViewer::Soc_VideoViewer(const int width, const int height, const std::string title)
	: width(width), height(height), title(title)
{
	
}

Soc_VideoViewer::~Soc_VideoViewer()
{
	cv::destroyAllWindows();
}

bool Soc_VideoViewer::Initialize()
{
	cv::namedWindow(title, cv::WINDOW_KEEPRATIO || cv::WINDOW_NORMAL);
	return true;
}

/// <summary>
/// Use this function to show an Mat frame
/// </summary>
/// <param name="frame"></param>
void Soc_VideoViewer::ShowVideo(cv::Mat& frame)
{
	if (!ImageUtil::CheckResolution(frame, width, height))
	{
		CORE_WARN("Frame to show is not the same size as the viewer, resizing to match viewer");
		cv::Mat res;
		ImageUtil::ResizeImage(frame, res, width, height);
		cv::imshow(title, res);
	}
	else
	{
		cv::imshow(title, frame);
	}
}

/// <summary>
/// Use this function to show an GpuMat frame
/// </summary>
/// <param name="frame"></param>
void Soc_VideoViewer::ShowVideo(cv::cuda::GpuMat& frame)
{
	if (!ImageUtil::CheckResolution(frame, width, height))
	{		
		CORE_WARN("Frame to show is not the same size as the viewer, resizing to match viewer");
		
		cv::cuda::GpuMat resGPU;
		ImageUtil::ResizeImage(frame, resGPU, width, height);
		cv::Mat resCPU;
		resGPU.download(resCPU);		
		cv::imshow(title, resCPU);		
		resGPU.release();
	}
	else
	{
		cv::Mat resCPU;
		frame.download(resCPU);
		cv::imshow(title, resCPU);
	}
}
