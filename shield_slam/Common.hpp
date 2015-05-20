#ifndef shield_slam_Common_hpp
#define shield_slam_Common_hpp

#include <opencv2/opencv.hpp>
#include <cassert>
#include <functional>
#include <memory>

using namespace cv;
using namespace std;

namespace vslam
{
    typedef std::vector<cv::KeyPoint> KeypointArray;
    
    typedef std::vector<cv::Point2f> PointArray;
    
    extern Mat camera_matrix, dist_coeff, img_size;
}

#endif
