#ifndef __shield_slam__Initializer__
#define __shield_slam__Initializer__

#include <opencv2/opencv.hpp>

#include "ORB.hpp"
#include "Common.hpp"
#include "MapPoint.hpp"

using namespace cv;
using namespace std;

#define SYMMETRIC_ERROR_SIGMA 1.0
#define SYMMETRIC_ERROR_TH 5.991

#define FUNDAMENTAL_ERROR_TH_SCORE 5.991
#define FUNDAMENTAL_ERROR_TH 3.841

#define HOMOGRAPHY_SELECTION_THRESHOLD 0.45

#define REPROJECTION_ERROR_TH 4.0
#define PARALLAX_MIN_DEGREES 1.0

#define TRIANGULATION_MIN_POINTS 50
#define TRIANGULATION_GOOD_POINTS_RATIO 0.9
#define TRIANGULATION_NORM_SCORE_H_TH 0.0
#define TRIANGULATION_NORM_SCORE_F_TH 0.7

namespace vslam
{
    
    class Initializer
    {
    public:
        
        Initializer();
        virtual ~Initializer() = default;
        
        void InitializeMap(vector<Mat>& init_imgs, vector<MapPoint>& map);
        
        float CheckHomography(PointArray& ref_keypoints, PointArray& tar_keypoints, Mat& H_ref2tar, vector<bool>& match_inliers, int& num_inliers);
        float CheckFundamental(PointArray& ref_keypoints, PointArray& tar_keypoints, Mat& F, vector<bool>& match_inliers, int& num_inliers);
        
        void CameraPoseHomography(Mat& H, Mat& pose);
        void CameraPoseFundamental(Mat& F, Mat& pose);
        
        bool ReconstructHomography(PointArray& ref_keypoints, PointArray& tar_keypoints, vector<DMatch>& matches, vector<bool>& inliers, int& num_inliers, Mat& H, Mat& R, Mat& t, vector<Point3f>& points, vector<bool>& triangulated_state);
        bool ReconstructFundamental(PointArray& ref_keypoints, PointArray& tar_keypoints, vector<DMatch>& matches, vector<bool>& inliers, int& num_inliers, Mat& F, Mat& R, Mat& t, vector<Point3f>& points, vector<bool>& triangulated_state);
        
        int CheckRt(Mat& R, Mat& t, const PointArray& ref_keypoints, const PointArray& tar_keypoints, const vector<bool>& inliers, const vector<DMatch>& matches, vector<Point3f>& point_cloud, float& max_parallax);
        float ScoreRt(vector<Mat>& p_R, vector<Mat>& p_t, const PointArray& ref_keypoints, const PointArray& tar_keypoints, const vector<bool>& inliers, const vector<DMatch>& matches, vector<Point3f>& best_point_cloud, float& best_parallax, int& best_trans_idx);
        
        void Triangulate(const KeyPoint& ref_keypoint, const KeyPoint& tar_keypoint, const Mat& P1, const Mat& P2, Mat& point_3D);
        
        void FilterInliers(PointArray& ref_keypoints, PointArray& tar_keypoints, vector<bool>& inliers, PointArray& ref_inliers, PointArray& tar_inliers);
        
    private:
        
        Ptr<ORB> orb_handler;
        
    protected:
        Mat R, t;
        vector<bool> triangulated_state;
    };
    
}

#endif /* defined(__shield_slam__Initializer__) */
