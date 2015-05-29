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
        KeyFrame(Mat& rot_mat, Mat& trans_mat, vector<MapPoint>& map, KeypointArray& total_kp, Mat& total_desc);
        virtual ~KeyFrame() = default;
        
        void SetRotation(Mat& rot) { R = rot.clone(); }
        void SetTranslation(Mat& trans) { t = t.clone(); }
        void SetLocalMap(vector<MapPoint>& map) { local_map = map; }
        
        Mat GetRotation(void) { return R; }
        Mat GetTranslation(void) { return t; }
        Mat GetDescriptors(void);
        vector<Point3f> Get3DPoints(void);
        vector<MapPoint> GetMap(void) { return local_map; }
        void GetKpDesc(PointArray& kp, Mat& desc);
        KeypointArray GetTrackedKeypoints(void);
        KeypointArray GetTotalKeypoints(void) { return orb_kp; }
        Mat GetTotalDescriptors(void) { return orb_desc; }
        int GetFrameCountSinceInsertion(void) { return insertion_frame_count; }
        
        void IncrementFrameCount(void) { insertion_frame_count++; }
        float ComputeMedianDepth(void);
        
    private:
        
    protected:
        Mat R, t;
        vector<MapPoint> local_map;
        KeypointArray orb_kp;
        Mat orb_desc;
        
        int insertion_frame_count;
    };
}

#endif /* defined(__shield_slam__KeyFrame__) */