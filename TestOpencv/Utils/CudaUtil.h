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

    static void InitializeBarrelGPUMap(
        const float distcofs[3],
        cv::Mat& cameraMatrix, cv::cuda::GpuMat& resultX, cv::cuda::GpuMat& resultY, bool releaseCameraMatrix = false,
        const int width = 4000, const int height = 4000) {

        cv::Mat distCoefs = (cv::Mat_<double>(1, 5) << distcofs[0], distcofs[1], distcofs[2], 0, 0);

        cv::Mat resultXCPU;
        cv::Mat resultYCPU;

        cv::initUndistortRectifyMap(
            cameraMatrix,
            distCoefs,
            cv::Mat(),
            cameraMatrix,
            cv::Size(width, height),
            CV_32FC1,
            resultXCPU,
            resultYCPU
        );

        resultX.upload(resultXCPU);
        resultY.upload(resultYCPU);

        resultXCPU.release();
        resultYCPU.release();
        distCoefs.release();

        if(releaseCameraMatrix)
            cameraMatrix.release();
    }
    
    static void Remap(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, cv::cuda::GpuMat& mapX, cv::cuda::GpuMat& mapY) {
        try {
            cv::cuda::remap(src, dst, mapX, mapY, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
        }
        catch (cv::Exception& e) {
			std::cout << e.what() << std::endl;
			std::cout << e.msg << std::endl;
		}
	}

    static void MakeWarpAffineMaps(
        const cv::Point2f srcPoints[3], const cv::Point2f dstPoints[3],
        cv::cuda::GpuMat& resultX, cv::cuda::GpuMat& resultY,
        const int imageWidth, const int imageHeight) {

        cv::Mat warpMatrixLeft;
        warpMatrixLeft = cv::getAffineTransform(srcPoints, dstPoints);

        cv::cuda::buildWarpAffineMaps(warpMatrixLeft, false, cv::Size(imageWidth, imageHeight), resultX, resultY);

        
        warpMatrixLeft.release();
    }

    static void MakeShiftYMap(int shift, int originalWidth, int originalHeight, cv::cuda::GpuMat &resultX, cv::cuda::GpuMat &resultY) {
        cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, 0, 0, 1, shift);

        cv::Size dsize = cv::Size(originalWidth, originalHeight + shift);
        try{
            cv::cuda::buildWarpAffineMaps(
				M, false, dsize, resultX, resultY
			);
		}
        catch (cv::Exception& e) {
			std::cout << e.what() << std::endl;
		}   
    
        M.release();
    }

};
 



