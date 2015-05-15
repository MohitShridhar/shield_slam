#include <opencv2/features2d/features2d.hpp>

#include "ORB.hpp"

using namespace cv;
using namespace std;

namespace vslam {
    
    ORB::ORB(int n_features, bool use_gpu)
    {
        if (!use_gpu)
        {
            detector = Ptr<FeatureDetector>(
                                            new GridAdaptedFeatureDetector(
                                                                           new cv::ORB(n_features, 1.2f, 8, 31, 0, 4, cv::ORB::HARRIS_SCORE, 31), n_features, GRID_CELL_ROWS, GRID_CELL_COLS));
            
            extractor = DescriptorExtractor::create("ORB");
            matcher = Ptr<BFMatcher>(new BFMatcher(NORM_HAMMING2, false));
        }
        else
        {
            //TODO: pending GPU implementation
        }
    }
    
    void ORB::ExtractFeatures(cv::Mat &img, KeypointArray &img_keypoints, cv::Mat &img_desc)
    {
        detector->detect(img, img_keypoints);
        extractor->compute(img, img_keypoints, img_desc);
    }
    
    void ORB::MatchFeatures(Mat &desc_ref, Mat &desc_tar, vector<cv::DMatch> matches, bool use_ratio_test)
    {
        if (desc_ref.empty() || desc_tar.empty())
        {
            CV_Error(0, "ORB::ExtractFeatures descriptors are empty");
        }
        
        // Brute-Force Matching:
        vector<vector<DMatch> > bf_matches_, good_matches_;
        matcher->knnMatch(desc_ref, desc_tar, bf_matches_, 2);
        
        matches.clear();
        
        for (int i=0; i<bf_matches_.size(); i++)
        {
            if (use_ratio_test)
            {
                if (bf_matches_[i][0].distance / bf_matches_[i][1].distance < KNN_RATIO_THRESHOLD)
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
    
}