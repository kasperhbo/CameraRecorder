//#include <iostream>
//

//
//int main() {
//    cv::VideoCapture clip("D:\\opencv\\assets\\compressed.mp4");
//
//    int fps = 1000 / 25;
//
//    if (!clip.isOpened()) {
//        std::cerr << "Failed to open video file." << std::endl;
//        return -1;
//    }
//
//
//    cv::Mat firstFrame;
//    clip.read(firstFrame);
//
//    //cv::Mat first_frame;
//    //clip.read(first_frame);
//
//    int ogw = firstFrame.cols;
//    int ogh = firstFrame.rows;
//
//    std::cout << ogw << std::endl;
//
//    std::string window_name = "Fake Camera Controller";
//    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
//
//
//    VideoController controller(ogw, ogh);
//
//    cv::Mat finalFrame;
//
//    while (true) {
//
//        if (!clip.read(finalFrame)) {
//            break;
//        }
//
//        finalFrame = controller.ProcessFrame(finalFrame);
//
//        //controller.ProcessFrame(finalFrame);
//
//        int key = cv::waitKey(20);
//
//        if (key == '+' || key == '=') {
//            controller.cameraZoom *= 1.1;
//        } else if (key == '-') {
//            controller.cameraZoom /= 1.1;
//        } else if (key == 'w') {
//            controller.yShift -= static_cast<int>(10);
//        } else if (key == 's') {
//            controller.yShift += static_cast<int>(10);
//        } else if (key == 'a') {
//            controller.xShift -= static_cast<int>(10);
//        } else if (key == 'd') {
//            controller.xShift += static_cast<int>(10);
//        } else if (key == 27) {
//            break;
//        }
//        
//        
//        cv::imshow(window_name, finalFrame);
//    }
//
//    finalFrame.release();
//    clip.release();
//    cv::destroyAllWindows();
//
//    return 0;
//}
//


#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <vector>
#include <chrono>
#include <iostream>

using namespace std;

cv::cuda::GpuMat initializeCudaMap(cv::cuda::GpuMat origina, ) {
	
	return 
}

int main() {

	bool hasCuda = cv::cuda::getCudaEnabledDeviceCount() > 0;
	std::cout << "Has CUDA: " << (hasCuda ? "true" : "false") << ", should be 1" << std::endl;

	if (!hasCuda) {
		std::cout << "CUDA is not available!" << std::endl;
		return -1;
	}

	const std::string clipSaveName = "D:\\opencv\\assets\\final.mp4";

	cv::VideoCapture clipLeft = cv::VideoCapture("D:\\opencv\\assets\\left0015.aví");
	cv::VideoCapture clipRight = cv::VideoCapture("D:\\opencv\\assets\\right0015.aví");

	cv::Mat firstFrame;

	clipLeft.read(firstFrame);
	int originalWidth = firstFrame.cols;
	int originalHeight = firstFrame.rows;
	firstFrame.release();

	
	cv::Mat map1Left, map2Left, map1Right, map2Right;
	double cofLeft[3] = { -0.199, -0.1300, -0.0150 };
	double cofRight[3] = { -0.4190, 0.0780, -0.0460 };


	cv::cuda::GpuMat map1LeftGPU, map2LeftGPU, map1RightGPU, map2RightGPU;

	map1Left.release();
	map2Left.release();
	map1Right.release();
	map2Right.release();



	cv::Mat frameLeft;
	cv::Mat frameRight;


	while (true)
	{
		//one of the clips is finished
		if (!clipLeft.read(frameLeft) || !clipRight.read(frameRight))
		{
			cout << "clip left or right is finished" << endl;
			break;
		}

		//


	}



	map1LeftGPU.release();
	map2LeftGPU.release();
	map1RightGPU.release();
	map2RightGPU.release();

	frameLeft.release();
	frameRight.release();
	
	clipLeft.release();
	clipRight.release();

}