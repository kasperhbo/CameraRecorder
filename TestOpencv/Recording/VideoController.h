#pragma once

#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafeatures2d.hpp>

class VideoController
{
public:
	VideoController(int initialWidth, int initialHeight);
	cv::Mat ProcessFrame(cv::Mat clip);


public:
	float cameraZoom;
	int xShift, yShift;

private:
	int initialWidth, initialHeight;
};

