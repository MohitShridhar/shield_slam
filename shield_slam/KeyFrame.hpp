#ifndef __shield_slam__KeyFrame__
#define __shield_slam__KeyFrame__

#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "Common.hpp"
#include "MapPoint.hpp"

using namespace cv;
using namespace std;

namespace vslam {
    
    class KeyFrame
    {
    public:
        KeyFrame();
        KeyFrame(Mat& rot_mat, Mat& trans_mat, vector<Point3f>& points, vector<Mat>& descriptors);
        KeyFrame(Mat& rot_mat, Mat& trans_mat, vector<MapPoint>& map);
        virtual ~KeyFrame() = default;
        
        Mat GetRotation(void) { return R; }
        Mat GetTranslation(void) { return t; }
        Mat GetDescriptors(void);
        vector<Point3f> Get3DPoints(void);
        
    private:
        
    protected:
        Mat R, t;
        vector<MapPoint> local_map;
        
    };
}

#endif /* defined(__shield_slam__KeyFrame__) */