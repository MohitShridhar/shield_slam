#ifndef __shield_slam__Tracking__
#define __shield_slam__Tracking__

#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "Common.hpp"
#include "MapPoint.hpp"
#include "KeyFrame.hpp"
#include "ORB.hpp"

#define TRIANGULATION_LS_ITERATIONS 10
#define TRIANGULATION_LS_EPSILON 0.0001

using namespace cv;
using namespace std;

namespace vslam {
    
    class Tracking
    {
    public:
        static void PosePnP(Ptr<ORB> orb_handler, const Mat& gray_frame, KeyFrame& kf, Mat& R, Mat& t);
        
        // Triangulation Functions:
        static void Triangulate(const KeyPoint &ref_keypoint, const KeyPoint &tar_keypoint,
                                const Mat &P1, const Mat &P2, Mat &point_3D);
        static Mat_<double> LinearLSTriangulation(const Point3d &u1, const Point3d &u2, const Mat &P1, const Mat &P2);
        static Matx31d IterativeLinearLSTriangulation(const Point3d &u1, const Point3d &u2, const Mat &P1, const Mat &P2);
    private:
        
    protected:
        

        
    };
}

#endif /* defined(__shield_slam__Tracking__) */