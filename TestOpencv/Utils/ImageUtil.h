#pragma once
#include <opencv2/core.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/shape/shape_transformer.hpp>
#include <opencv2/cudacodec.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>

#include "CudaUtil.h"

class ImageUtil
{
public:
	static void ResizeImage(cv::Mat& src, cv::Mat& dst, const int width, const int height) {
		cv::resize(src, dst, cv::Size(width, height));
	}

	static void ResizeImage(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, const int width, const int height) {
		cv::cuda::resize(src, dst, cv::Size(width, height));
	}

	/// <summary>
	/// Concatenate two images horizontally
	/// </summary>
	/// <param name="left"></param>
	/// <param name="right"></param>
	/// <param name="result"></param>
	static void HConcat(const cv::Mat left, const cv::Mat right, cv::Mat& result) {
		cv::hconcat(left, right, result);
	}

	/// <summary>
	/// Concatenate two images horizontally
	/// </summary>
	/// <param name="left"></param>
	/// <param name="right"></param>
	/// <param name="result"></param>
	static void HConcat(const cv::cuda::GpuMat left, const cv::cuda::GpuMat right, cv::cuda::GpuMat& result) {
		int size_cols = left.cols + right.cols;
		int size_rows = std::max(left.rows, right.rows);
		cv::cuda::GpuMat hconcat(size_rows, size_cols, left.type());
		left.copyTo(hconcat(cv::Rect(0, 0, left.cols, left.rows)));
		right.copyTo(hconcat(cv::Rect(left.cols, 0, right.cols, right.rows)));

		result = hconcat.clone();

		hconcat.release();
	}

	/// <summary>
	/// Use this function to compare the resolution of the video frame with the resolution of the video viewer
	/// </summary>
	/// <param name="frame"></param>
	/// <returns>is same res</returns>
	static bool CheckResolution(cv::Mat& frame, int width, int height)
	{
		return frame.cols == width && frame.rows == height;
	}


	/// <summary>
	/// Use this function to compare the resolution of the video frame with the resolution of the video viewer
	/// </summary>
	/// <param name="frame"></param>
	/// <returns>is same res</returns>
	static bool CheckResolution(cv::cuda::GpuMat& frame, int width, int height)
	{
		return frame.cols == width && frame.rows == height;
	}
};