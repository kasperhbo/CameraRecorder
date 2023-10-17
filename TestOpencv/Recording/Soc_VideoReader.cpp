#include "Soc_VideoReader.h"

#include "../Utils/Log.h"

using namespace TestOpencv;

Soc_VideoReader::Soc_VideoReader(const cv::String locationLeft, const cv::String locationRight)
	: locationLeft(locationLeft), locationRight(locationRight)
{
	
}

Soc_VideoReader::~Soc_VideoReader()
{
	/*clipLeft.release();
	clipRight.release();*/
	d_readerLeft.release();
	d_readerRight.release();
}

bool Soc_VideoReader::Initialize() {
	try {
		CORE_CRITICAL("Opening video reader");
		d_readerLeft = cv::cudacodec::createVideoReader(locationLeft);
	}
	catch (const cv::Exception e) {
		CORE_ERROR("Error opening video reader: {}", e.msg);
		return false;
	}

	try {
		CORE_CRITICAL("Opening video reader");
		d_readerRight = cv::cudacodec::createVideoReader(locationRight);

	}
	catch (const cv::Exception e) {
		CORE_ERROR("Error opening video reader: {}", e.msg);
		return false;
	}
	return true;
}


void Soc_VideoReader::ResetClips()
{
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
	try {
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
	catch (const cv::Exception e) {
		CORE_ERROR("Error reading frame: {}", e.msg);
		return false;
	}
}
