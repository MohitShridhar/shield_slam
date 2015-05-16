#include <opencv2/features2d/features2d.hpp>

#include "ORB.hpp"

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
                            PointArray& ref_matches, PointArray& tar_matches,
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
        
        for (int i=0; i<bf_matches_.size(); i++)
        {
            if (use_ratio_test)
            {
                if (bf_matches_[i][0].distance / bf_matches_[i][1].distance < KNN_RATIO_THRESHOLD)
                {
                    matches.push_back(bf_matches_[i][0]);
                    
                    ref_matches.push_back(ref_keypoints[bf_matches_[i][0].queryIdx].pt);
                    tar_matches.push_back(tar_keypoints[bf_matches_[i][0].trainIdx].pt);
                }
            }
            else
            {
                matches.push_back(bf_matches_[i][0]);
            }
        }
    }
    
    void ORB::DetectAndMatch(Mat &img_ref, Mat &img_tar, vector<cv::DMatch> &matches,
                             PointArray& ref_matches, PointArray& tar_matches)
    {
        KeypointArray keypoints_ref_, keypoints_tar_;
        Mat desc_ref_, desc_tar_;
        
        ExtractFeatures(img_ref, keypoints_ref_, desc_ref_);
        ExtractFeatures(img_tar, keypoints_tar_, desc_tar_);
        
        MatchFeatures(desc_ref_, desc_tar_, matches, keypoints_ref_, keypoints_tar_, ref_matches, tar_matches);
        
        // Debug:
        //        cout << "Number of matches " << matches.size() << endl;
        //        Mat debug_draw_img;
        //        drawMatches(img_ref, keypoints_ref_, img_tar, keypoints_tar_, matches, debug_draw_img);
        //
        //        imshow("init orb matches", debug_draw_img);
        //        waitKey(0);
        //------
    }
    
}
