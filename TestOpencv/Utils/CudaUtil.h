#pragma once

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

#include <iostream>

class CudaUtil {
private:
    static std::string currentDeviceName;

public:
    static bool IsCudaAvailable() {
        return cv::cuda::getCudaEnabledDeviceCount() > 0;
    }

    static int GetCudaDeviceCount() {      
        return cv::cuda::getCudaEnabledDeviceCount();
    }

    static std::string GetCudaBuildInformation() {
        return cv::getBuildInformation();
    }

    static void UploadToGPU(cv::Mat& src, cv::cuda::GpuMat& dst) {
        dst.upload(src);
    }

    static void DownloadFromGPU(cv::cuda::GpuMat& src, cv::Mat& dst) {
        src.download(dst);
    }

    static void ReleaseGPUData(cv::cuda::GpuMat& data) {
        data.release();
    }

    static std::string GetCudaDeviceName() {
        return currentDeviceName;
    }

    static cv::cuda::DeviceInfo GetCudaDeviceProperties(int deviceId) {
        cv::cuda::DeviceInfo deviceInfo(deviceId);
        return deviceInfo;
    }

    static void CustomHConcat(const cv::cuda::GpuMat src1, const cv::cuda::GpuMat src2, cv::cuda::GpuMat& result)
    {
        int size_cols = src1.cols + src2.cols;
        int size_rows = std::max(src1.rows, src2.rows);
        cv::cuda::GpuMat hconcat(size_rows, size_cols, src1.type());
        src1.copyTo(hconcat(cv::Rect(0, 0, src1.cols, src1.rows)));
        src2.copyTo(hconcat(cv::Rect(src1.cols, 0, src2.cols, src2.rows)));

        result = hconcat.clone();
    }

};
 



