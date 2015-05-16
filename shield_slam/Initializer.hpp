#ifndef __shield_slam__Initializer__
#define __shield_slam__Initializer__

#include <opencv2/opencv.hpp>

#include "ORB.hpp"
#include "Common.hpp"

using namespace cv;
using namespace std;

#define SYMMETRIC_ERROR_SIGMA 1.0
#define SYMMETRIC_ERROR_TH 5.991

#define FUNDAMENTAL_ERROR_TH_SCORE 5.991
#define FUNDAMENTAL_ERROR_TH 3.841

#define HOMOGRAPHY_SELECTION_THRESHOLD 0.45

namespace vslam
{
    
    class Initializer
    {
    public:
        
        Initializer();
        virtual ~Initializer() = default;
        
        void InitializeMap(vector<Mat>& init_imgs);
        float CheckHomography(PointArray& ref_keypoints, PointArray& tar_keypoints, Mat& H_ref2tar, vector<bool>& match_inliers);
        float CheckFundamental(PointArray& ref_keypoints, PointArray& tar_keypoints, Mat& F, vector<bool>& match_inliers);
        
    private:
        
        Ptr<ORB> orb_handler;
    };
    
}

#endif /* defined(__shield_slam__Initializer__) */
