#include "Soc_VideoReader.h"

#include "../Log.h"

using namespace TestOpencv;

Soc_VideoReader::Soc_VideoReader(const std::string locationLeft, const std::string locationRight)
	: locationLeft(locationLeft), locationRight(locationRight)
{
	try {
		d_readerLeft = cv::cudacodec::createVideoReader(locationLeft);
	}
	catch (const cv::Exception e) {
		CORE_ERROR("Error opening video reader: {}", e.msg); 
	}

	try {
		d_readerRight = cv::cudacodec::createVideoReader(locationRight);
	}
	catch (const cv::Exception e) {
		CORE_ERROR("Error opening video reader: {}", e.msg);
	}		
}

Soc_VideoReader::~Soc_VideoReader()
{
	/*clipLeft.release();
	clipRight.release();*/
	d_readerLeft.release();
	d_readerRight.release();
}

void Soc_VideoReader::ResetClips()
{

	/*clipLeft.release();
clipRight.release();*/
	d_readerLeft.release();
	d_readerRight.release();
	try {
		d_readerLeft = cv::cudacodec::createVideoReader(locationLeft);
	}
	catch (const cv::Exception e) {
		CORE_ERROR("Error opening video reader: {}", e.msg);
	}

	try {
		d_readerRight = cv::cudacodec::createVideoReader(locationRight);
	}
	catch (const cv::Exception e) {
		CORE_ERROR("Error opening video reader: {}", e.msg);
	}

	CORE_INFO("Clips reset");
}

bool Soc_VideoReader::Read(cv::cuda::GpuMat& resultLeft, cv::cuda::GpuMat& resultRight)
{
	if (!d_readerLeft->nextFrame(resultLeft))
	{
		CORE_INFO("Left clip is finished");
		return false;
	}
		
	if (!d_readerRight->nextFrame(resultRight))
	{
		CORE_INFO("Left clip is finished");
		return false;
	}

	return true;
}
