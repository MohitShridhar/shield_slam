#ifndef __shield_slam__Tracking__
#define __shield_slam__Tracking__

#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <math.h>
#include <limits>
#include <map>

#include "../ss/Common.hpp"
#include "../ss/KeyFrame.hpp"
#include "../ss/MapPoint.hpp"
#include "../ss/ORB.hpp"

#define TRIANGULATION_LS_ITERATIONS 10
#define TRIANGULATION_LS_EPSILON 0.0001

#define REPROJECTION_ERROR_TH 20000.0
#define REPROJECTION_ERROR_CHI 5.991
#define TRIANGULATION_MIN_POINTS 4

#define KEYFRAME_MIN_KEYPOINTS 50
#define KEYFRAME_MIN_MATCH_RATIO 0.7
#define KEYFRAME_MAX_FRAME_COUNT_SINCE_INSERTION 10

#define ORB_SCALE_FACTOR 1.2

using namespace cv;
using namespace std;

namespace vslam {
    
    class Tracking
    {
    public:
        static bool TrackMap(const Mat& gray_frame, vector<KeyFrame>& keyframes, Mat& R, Mat& t, bool& new_kf_added);
        static bool NewKeyFrame(KeyFrame &kf, Mat &R1, Mat &R2, Mat &t1, Mat &t2,
                                KeypointArray &kp1, KeypointArray &kp2,
                                Mat& ref_desc, Mat& tar_desc, vector<DMatch>& matches_2D_3D,
                                Mat& pnp_inliers, double max_val, vector<Point3f>& prev_pc);
        
        static void SetOrbHandler(Ptr<ORB> handler)  { orb_handler = handler; }
        static void SetInitScale(double scale)  { init_scale = scale; }
        
        static bool CheckDistEpipolarLine(const KeyPoint &kp1,const KeyPoint &kp2,const Mat &F);
        
        // Triangulation Functions:
        static void AlternateTriangulate(const KeyPoint &ref_keypoint, const KeyPoint &tar_keypoint, const Mat &P1, const Mat &P2, Mat &point_3D);
        static void Triangulate(const KeyPoint &ref_keypoint, const KeyPoint &tar_keypoint,
                                const Mat &P1, const Mat &P2, Mat &point_3D);
        static void TriangulateAlt(const KeyPoint &ref_keypoint, const KeyPoint &tar_keypoint,
                                const Mat &P1, const Mat &P2, Mat &point_3D);
        static Mat_<double> LinearLSTriangulation(const Point3d &u1, const Point3d &u2, const Mat &P1, const Mat &P2);
        static Matx31d IterativeLinearLSTriangulation(const Point3d &u1, const Point3d &u2, const Mat &P1, const Mat &P2);
        
        static double FindLinearScale(Mat& R, Mat& t, vector<Point2f>& image_points, vector<Point3f>& object_points);
        
    private:
        static bool NeedsNewKeyframe(KeyFrame& kf, int num_kf_kp, int num_tar_kp, int num_kf_matches);
        static void FilterPnPInliers(vector<Point3f>& object_points, vector<Point2f>& image_points, Mat& inliers);
        static void Normalize3DPoints(vector<Point3f>& input_points, vector<Point3f>& norm_points);
        
    protected:
        static Ptr<ORB> orb_handler;
        static double init_scale;
        
        static bool has_scale_init;
    };
}

#endif /* defined(__shield_slam__Tracking__) */