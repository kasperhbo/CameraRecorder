#include "Soc_VideoWriter.h"

#include "../Utils/CudaUtil.h"
#include "../Utils/Log.h"

using namespace TestOpencv;

Soc_VideoWriter::Soc_VideoWriter(cv::String location, int width, int height, int fps) : 
	location(location), width(width), height(height), fps(fps)
{	
	
}

Soc_VideoWriter::~Soc_VideoWriter()
{
	CORE_INFO("Releasing writer");
	d_writer.release();
}


bool Soc_VideoWriter::Initialize() 
{
	cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());

	CORE_INFO("Initializing video writer");

	if (d_writer.empty())
	{
		try {
			d_writer = cv::cudacodec::createVideoWriter(
				location, cv::Size(width, height), cv::cudacodec::Codec::H264, fps,
				cv::cudacodec::ColorFormat::BGRA, 0, stream);

			CORE_INFO("Writing to {} is finshed setting up", location);
			return true;
		}
		catch (const cv::Exception e) {
			CORE_ERROR("Error creating video writer: {}", e.msg);
			return false;
		}
	}
	else
	{
		CORE_WARN("Writer is not empty, make sure to release it before initializing");
		return false;
	}
}


/// <summary>
/// Write the two frames to the video as one frame
/// </summary>
/// <param name="frameLeft"></param>
/// <param name="frameRight"></param>
/// <param name="result"></param>
void Soc_VideoWriter::Write(cv::cuda::GpuMat& frameToShow)
{
	if (frameToShow.empty())
	{
		CORE_ERROR("Frame to show is empty");
		return;
	}

	if (d_writer.empty())
	{
		CORE_ERROR("Writer is empty, make sure its initialized");
		return;
	}

	if (!ImageUtil::CheckResolution(frameToShow, width, height))
	{
		CORE_WARN("Frame to show is not the same size as the writer, resizing to match writer");

		cv::cuda::GpuMat resGPU;
		ImageUtil::ResizeImage(frameToShow, resGPU, width, height);

		try {
			d_writer->write(resGPU);
		}
		catch (const cv::Exception e) {
			CORE_ERROR("Error writing frame: {}", e.msg);
			return;
		}
		resGPU.release();



	}
	else {
		try {
			d_writer->write(frameToShow);
		}
		catch (const cv::Exception e) {
			CORE_ERROR("Error writing frame: {}", e.msg);
			return;
		}
	}
}

