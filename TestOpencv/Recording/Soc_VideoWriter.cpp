#include "Soc_VideoWriter.h"

#include "../Utils/CudaUtil.h"
#include "../Utils/Log.h"

using namespace TestOpencv;

Soc_VideoWriter::Soc_VideoWriter(const cv::String  location, double fps, int widthres, int heightres)  : widthres(widthres), heightres(heightres)
{	
	cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());

	cv::VideoWriter writer;
	//cv::Ptr<cv::cudacodec::VideoWriter> d_writer;

	cv::Mat frame;
	cv::cuda::GpuMat d_frame;

	if (d_writer.empty())
	{
		
		d_writer = cv::cudacodec::createVideoWriter(
			location, cv::Size(widthres, heightres), cv::cudacodec::Codec::H264, fps, 
			cv::cudacodec::ColorFormat::BGRA, 0, stream);

		CORE_INFO("Writing to {}", location);
	}
}

Soc_VideoWriter::~Soc_VideoWriter()
{
	d_writer.release();		
}


/// <summary>
/// Write the two frames to the video as one frame
/// </summary>
/// <param name="frameLeft"></param>
/// <param name="frameRight"></param>
/// <param name="result"></param>
void Soc_VideoWriter::Write(const cv::cuda::GpuMat frameLeft, const cv::cuda::GpuMat frameRight, cv::cuda::GpuMat& result)
{
 
	CudaUtil::CustomHConcat(frameLeft, frameRight, result);

	//CudaUtils::CustomHConcat(frameLeft, frameRight, result);

	cv::cuda::resize(result, result, cv::Size(widthres, heightres));

	//cv::Mat resCPU;
	try {

		d_writer->write(result);
		//std::cout << "Writing frame done" << std::endl;
	}
	catch (const cv::Exception e) {
		CORE_ERROR("Error writing frame: {}", e.msg);
	}
}

