#include "../ss/ORB.hpp"

#include <opencv2/features2d/features2d.hpp>


using namespace cv;
using namespace std;

namespace vslam {
    
    ORB::ORB(int n_features, bool use_gpu)
    {
        detector = Ptr<FeatureDetector>(
                                        new GridAdaptedFeatureDetector(
                                                                       new cv::ORB(n_features, 1.2f, 8, 31, 0, 4, cv::ORB::HARRIS_SCORE, 31), n_features, GRID_CELL_ROWS, GRID_CELL_COLS));
        
        extractor = DescriptorExtractor::create("ORB");
        matcher = Ptr<BFMatcher>(new BFMatcher(NORM_HAMMING2, false));
        
        // TODO: GPU Implementation
    }
    
    void ORB::ExtractFeatures(cv::Mat &img, KeypointArray &img_keypoints, cv::Mat &img_desc)
    {
        detector->detect(img, img_keypoints);
        extractor->compute(img, img_keypoints, img_desc);
    }
    
    
    void ORB::MatchFeatures(Mat &desc_ref, Mat &desc_tar, vector<cv::DMatch> &matches,
                            KeypointArray& ref_keypoints, KeypointArray& tar_keypoints,
                            PointArray& ref_matches, PointArray& tar_matches, Mat &matched_tar_desc,
                            bool use_ratio_test)
    {
        if (desc_ref.empty() || desc_tar.empty())
        {
            CV_Error(0, "ORB::ExtractFeatures descriptors are empty");
        }
        
        // Brute-Force Matching:
        vector<vector<DMatch> > bf_matches_, good_matches_;
        matcher->knnMatch(desc_ref, desc_tar, bf_matches_, 2);
        
        matches.clear();
        ref_matches.clear();
        tar_matches.clear();
        
        vector<Mat> kp_desc;
        for (int i=0; i<bf_matches_.size(); i++)
        {
            if (use_ratio_test)
            {
                if (bf_matches_[i][0].distance / bf_matches_[i][1].distance < KNN_RATIO_INIT_THRESHOLD)
                {
                    matches.push_back(bf_matches_[i][0]);
                    
                    ref_matches.push_back(ref_keypoints[bf_matches_[i][0].queryIdx].pt);
                    tar_matches.push_back(tar_keypoints[bf_matches_[i][0].trainIdx].pt);
                    
                    kp_desc.push_back(desc_tar.row(bf_matches_[i][0].trainIdx));
                }
            }
            else
            {
                matches.push_back(bf_matches_[i][0]);
            }
        }
        
        matched_tar_desc = Mat::zeros((int)kp_desc.size(), kp_desc.at(0).cols, CV_8U);
        for (int i=0; i<kp_desc.size(); i++)
        {
            kp_desc.at(i).copyTo(matched_tar_desc.row(i));
        }
    }
    
    void ORB::MatchFeatures(Mat& desc_ref, Mat& desc_tar, vector<DMatch>& matches, bool use_ratio_test)
    {
        if (desc_ref.empty() || desc_tar.empty())
        {
            CV_Error(0, "ORB::ExtractFeatures descriptors are empty");
        }
        
        // Brute-Force Matching:
        vector<vector<DMatch> > bf_matches_;
        matcher->knnMatch(desc_ref, desc_tar, bf_matches_, 2);
        
        matches.clear();
        
        for (int i=0; i<bf_matches_.size(); i++)
        {
            if (use_ratio_test)
            {                
                if (bf_matches_[i][0].distance / bf_matches_[i][1].distance < KNN_RATIO_TRACKING_THRESHOLD)
                {
                    matches.push_back(bf_matches_[i][0]);
                }
            }
            else
            {
                matches.push_back(bf_matches_[i][0]);
            }
        }

    }
    
    void ORB::DetectAndMatch(Mat &img_ref, Mat &img_tar, vector<cv::DMatch> &matches,
                             PointArray& ref_matches, PointArray& tar_matches, Mat &matched_tar_desc,
                             KeypointArray& ref_keypoints, KeypointArray& tar_keypoints,
                             Mat& ref_desc, Mat& tar_desc)
    {
        ExtractFeatures(img_ref, ref_keypoints, ref_desc);
        ExtractFeatures(img_tar, tar_keypoints, tar_desc);
        
        MatchFeatures(ref_desc, tar_desc, matches, ref_keypoints, tar_keypoints, ref_matches, tar_matches, matched_tar_desc);
        
        // Debug:
        // cout << "Number of matches " << matches.size() << endl;
        // Mat debug_draw_img;
        // drawMatches(img_ref, ref_keypoints, img_tar, tar_keypoints, matches, debug_draw_img);
        
        // imshow("init orb matches", debug_draw_img);
        // waitKey(0);
        //------
    }
    
    
}
