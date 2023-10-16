#include "VideoController.h"

VideoController::VideoController(int initialWidth, int initialHeight)
    : initialWidth(initialWidth), initialHeight(initialHeight),
    cameraZoom(1.0), xShift(0), yShift(0) {

    std::cout << initialWidth << std::endl;
    std::cout << initialHeight << std::endl;

}

cv::Mat VideoController::ProcessFrame(cv::Mat frame)
{
    int roi_h = static_cast<int>(initialHeight / cameraZoom);
    int roi_w = static_cast<int>(initialWidth / cameraZoom);
        
    int x1 = std::min(std::max(0, initialWidth / 2 - roi_w / 2 + xShift), initialWidth - roi_w);
    int y1 = std::min(std::max(0, initialHeight / 2 - roi_h / 2 + yShift), initialHeight - roi_h);

    int x2 = x1 + roi_w;
    int y2 = y1 + roi_h;

    // Crop and resize the frame
    return frame(cv::Rect(x1, y1, roi_w, roi_h));
}


