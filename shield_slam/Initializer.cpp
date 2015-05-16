#include "Initializer.hpp"

using namespace cv;
using namespace std;

namespace vslam
{
    
    Initializer::Initializer()
    {
        orb_handler = new ORB(500, false);
    }
    
    void Initializer::InitializeMap(vector<cv::Mat> &init_imgs)
    {
        Mat img_ref, img_tar;
        
        // TODO: this is assuming init_imgs has only two images. Check for better initialization
        img_ref = init_imgs.at(0);
        img_tar = init_imgs.at(1);
        
        // Match ORB Features:
        vector<DMatch> matches;
        PointArray ref_matches, tar_matches;
        orb_handler->DetectAndMatch(img_ref, img_tar, matches, ref_matches, tar_matches);
        
        /*
        // Undistort key points using camera intrinsics:
        PointArray undist_ref_matches, undist_tar_matches;
        undistort(ref_matches, undist_ref_matches, camera_matrix, dist_coeff);
        undistort(tar_matches, undist_tar_matches, camera_matrix, dist_coeff);
        */
        
        // Compute homography and fundamental matrices:
        Mat H = findHomography(ref_matches, tar_matches, CV_RANSAC, 3);
        Mat F = findFundamentalMat(ref_matches, tar_matches, CV_FM_RANSAC, 3, 0.99);
        
        vector<bool> h_inliers, f_inliers;
        float SH = CheckHomography(ref_matches, tar_matches, H, h_inliers);
        float SF = CheckFundamental(ref_matches, tar_matches, F, f_inliers);
        
        float RH = SH / (SH + SF);

        if (RH > HOMOGRAPHY_SELECTION_THRESHOLD)
        {
            
        }
        else
        {
            
        }
        
    }
    
    float Initializer::CheckHomography(PointArray& ref_keypoints, PointArray& tar_keypoints, Mat &H_ref2tar, vector<bool> &match_inliers)
    {
        float score = 0;
        Mat H_tar2ref = H_ref2tar.inv();
        
        const float inv_sigma_square = 1.0 / (SYMMETRIC_ERROR_SIGMA * SYMMETRIC_ERROR_SIGMA);
        match_inliers.resize(ref_keypoints.size());
        
        assert(ref_keypoints.size() == tar_keypoints.size());
        for(int i=0; i<ref_keypoints.size(); i++)
        {
            bool is_inlier = true;
            
            const float x1 = ref_keypoints[i].x;
            const float y1 = ref_keypoints[i].y;
            const float x2 = tar_keypoints[i].x;
            const float y2 = tar_keypoints[i].y;
            
            // Reproject tar keypoints to ref keypoints:
            Mat x2_y2 = (Mat_<double>(3, 1) << x2, y2, 1.0f);
            Mat reproj_x1_y1 = H_tar2ref * x2_y2;
            
            const float reproj_w1 = 1.0 / reproj_x1_y1.at<double>(2, 0);
            const float reproj_x1 = reproj_x1_y1.at<double>(0, 0) * reproj_w1;
            const float reproj_y1 = reproj_x1_y1.at<double>(1, 0) * reproj_w1;
            
            // Euclidean distance between 2D points:
            const float ref_square_dist = (x1-reproj_x1)*(x1-reproj_x1) + (y1-reproj_y1)*(y1-reproj_y1);
            const float ref_chi_square = ref_square_dist * inv_sigma_square;
            
            if (ref_chi_square > SYMMETRIC_ERROR_TH)
            {
                is_inlier = false;
            }
            else
            {
                score += SYMMETRIC_ERROR_TH - ref_chi_square;
            }
            
            // Reproject ref keypoints to tar keypoints;
            Mat x1_y1 = (Mat_<double>(3, 1) << x1, y1, 1.0f);
            Mat reproj_x2_y2 = H_ref2tar * x1_y1;
            
            const float reproj_w2 = 1.0 / reproj_x2_y2.at<double>(2, 0);
            const float reproj_x2 = reproj_x2_y2.at<double>(0, 0) * reproj_w2;
            const float reproj_y2 = reproj_x2_y2.at<double>(1, 0) * reproj_w2;
            
            // Euclidean distance between 2D points:
            const float tar_square_dist = (x2-reproj_x2)*(x2-reproj_x2) + (y2-reproj_y2)*(y2-reproj_y2);
            const float tar_chi_square = tar_square_dist * inv_sigma_square;
            
            if (tar_square_dist > SYMMETRIC_ERROR_TH)
            {
                is_inlier = false;
            }
            else
            {
                score += SYMMETRIC_ERROR_TH - tar_chi_square;
            }
            
            
            // Update inlier status:
            if (is_inlier)
            {
                match_inliers[i] = true;
            }
            else
            {
                match_inliers[i] = false;
            }
            
        }
        
        return score;
    }
    
    float Initializer::CheckFundamental(PointArray &ref_keypoints, PointArray &tar_keypoints, Mat &F, vector<bool> &match_inliers)
    {
        float score = 0;
        
        const float inv_sigma_square = 1.0 / (SYMMETRIC_ERROR_SIGMA * SYMMETRIC_ERROR_SIGMA);
        match_inliers.resize(ref_keypoints.size());
        
        assert(ref_keypoints.size() == tar_keypoints.size());
        for (int i=0; i<ref_keypoints.size(); i++)
        {
            bool is_inliner = true;
            
            const float x1 = ref_keypoints[i].x;
            const float y1 = ref_keypoints[i].y;
            const float x2 = tar_keypoints[i].x;
            const float y2 = tar_keypoints[i].y;
            
            Mat x1_y1 = (Mat_<double>(3, 1) << x1, y1, 1.0f);
            Mat x2_y2 = (Mat_<double>(3, 1) << x2, y2, 1.0f);
            Mat x1_y1_tp = x1_y1.t();
            Mat x2_y2_tp = x2_y2.t();
            
            // Project ref keypoints to target keypoints (aT * F * a'):
            Mat F_alpha_prime = F * x1_y1;
            Mat alpha_tp_F_alpha_prime = x2_y2_tp * F_alpha_prime;
            
            const float ref_square_dist = (alpha_tp_F_alpha_prime.at<double>(0, 0) * alpha_tp_F_alpha_prime.at<double>(0, 0)) /             ( F_alpha_prime.at<double>(0, 0) * F_alpha_prime.at<double>(0, 0) + F_alpha_prime.at<double>(0, 1) * F_alpha_prime.at<double>(0, 1) );
            
            const float ref_chi_square = ref_square_dist * inv_sigma_square;
            
            if (ref_chi_square > FUNDAMENTAL_ERROR_TH)
            {
                is_inliner = false;
            }
            else
            {
                score += FUNDAMENTAL_ERROR_TH_SCORE - ref_chi_square;
            }
            
            // Project tar keypoints to ref keypoints (a'T * F * a):
            Mat F_alpha = F * x2_y2;
            Mat alpha_prime_tp_F_alpha = x1_y1_tp * F_alpha;
            
            const float tar_square_dist = (alpha_prime_tp_F_alpha.at<double>(0, 0) * alpha_prime_tp_F_alpha.at<double>(0, 0)) /
            (F_alpha.at<double>(0, 0) * F_alpha.at<double>(0, 0) + F_alpha.at<double>(0, 1) * F_alpha.at<double>(0, 1));
            
            const float tar_chi_square = tar_square_dist * inv_sigma_square;
            
            if (tar_chi_square > FUNDAMENTAL_ERROR_TH)
            {
                is_inliner = false;
            }
            else
            {
                score += FUNDAMENTAL_ERROR_TH_SCORE - tar_chi_square;
            }
            
            if (is_inliner)
            {
                match_inliers[i] = true;
            }
            else
            {
                match_inliers[i] = false;
            }
            
        }
        
        return score;
    }
    
}
