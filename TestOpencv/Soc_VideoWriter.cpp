#include "Soc_VideoWriter.h"
	

SocVideoWriter::SocVideoWriter(const cv::String  location, double fps, int widthres, int heightres)  : widthres(widthres), heightres(heightres)
{
	cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());

	cv::VideoWriter writer;
	//cv::Ptr<cv::cudacodec::VideoWriter> d_writer;

	cv::Mat frame;
	cv::cuda::GpuMat d_frame;

	if (d_writer.empty())
	{
		//std::cout << "Open CUDA Writer" << std::endl;
		//const cv::String outputFilename = "output_gpu.h264";
		d_writer = cv::cudacodec::createVideoWriter(location, cv::Size(widthres, heightres), cv::cudacodec::Codec::H264, fps, cv::cudacodec::ColorFormat::BGR, 0, stream);
		std::cout << "Writing to " << location << std::endl;
	}
}

SocVideoWriter::~SocVideoWriter()
{
	d_writer.release();		
}


/// <summary>
/// Write the two frames to the video as one frame
/// </summary>
/// <param name="frameLeft"></param>
/// <param name="frameRight"></param>
/// <param name="result"></param>
void SocVideoWriter::Write(const cv::cuda::GpuMat frameLeft, const cv::cuda::GpuMat frameRight, cv::cuda::GpuMat& result)
{
 
	CustomHConcat(frameLeft, frameRight, result);

	/*std::cout << "Writing frame" << std::endl;
	std::cout << "result width  size: " << / << std::endl;
	std::cout << "result width  size: " << heightres << std::endl;*/

	cv::cuda::resize(result, result, cv::Size(widthres, heightres));

	//cv::Mat resCPU;
	try {
		d_writer->write(result);
		std::cout << "Writing frame done" << std::endl;
	}
	catch (const cv::Exception e) {
		std::cout << "Error writing frame" << std::endl;
		std::cout << e.msg << std::endl;
	}
	//result.download(resCPU);

	//writer.write(resCPU);		
	
}


/// <summary>
/// Stitch two images together horizontally
/// </summary>
/// <param name="src1">		</param>
/// <param name="src2">		</param>
/// <param name="result">	</param>
void SocVideoWriter::CustomHConcat(const cv::cuda::GpuMat src1, const cv::cuda::GpuMat src2, cv::cuda::GpuMat& result)
{
	int size_cols = src1.cols + src2.cols;
	int size_rows = std::max(src1.rows, src2.rows);
	cv::cuda::GpuMat hconcat(size_rows, size_cols, src1.type());
	src1.copyTo(hconcat(cv::Rect(0, 0, src1.cols, src1.rows)));
	src2.copyTo(hconcat(cv::Rect(src1.cols, 0, src2.cols, src2.rows)));

	result = hconcat.clone();
}
