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
        KeyFrame(Mat& rot_mat, Mat& trans_mat, vector<MapPoint>& map, KeypointArray& total_kp, Mat& total_desc);
        virtual ~KeyFrame() = default;
        
        Mat GetRotation(void) { return R; }
        Mat GetTranslation(void) { return t; }
        Mat GetDescriptors(void);
        vector<Point3f> Get3DPoints(void);
        vector<MapPoint> GetMap(void) { return local_map; }
        void GetKpDesc(PointArray& kp, Mat& desc);
        KeypointArray GetTotalKeypoints(void) { return orb_kp; }
        Mat GetTotalDescriptors(void) { return orb_desc; }
        
    private:
        
    protected:
        Mat R, t;
        vector<MapPoint> local_map;
        KeypointArray orb_kp;
        Mat orb_desc;
    };
}

#endif /* defined(__shield_slam__KeyFrame__) */