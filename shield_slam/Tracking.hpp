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

#define REPROJECTION_ERROR_TH 10.0
#define TRIANGULATION_MIN_POINTS 50

using namespace cv;
using namespace std;

namespace vslam {
    
    class Tracking
    {
    public:
        static void TrackPnP(Ptr<ORB> orb_handler, const Mat& gray_frame, KeyFrame& kf, Mat& R, Mat& t, bool add_new_kf);
        static bool NewKeyFrame(KeyFrame &kf, Mat &R1, Mat &R2, Mat &t1, Mat &t2,
                                KeypointArray &kp1, KeypointArray &kp2,
                                Mat& ref_desc, Mat& tar_desc, vector<DMatch>& matches);
        
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