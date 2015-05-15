#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "Common.hpp"

#define GRID_CELL_ROWS 1
#define GRID_CELL_COLS 1

#define KNN_RATIO_THRESHOLD 0.7

using namespace cv;
using namespace std;

namespace vslam {
    
    class ORB
    {
        
    public:
        
        ORB(int n_features = 500, bool use_gpu = false);
        virtual ~ORB() = default;
        
        void ExtractFeatures (Mat& img, KeypointArray& img_keypoints, Mat& img_desc);
        void MatchFeatures (Mat& desc_ref, Mat& desc_tar, vector<DMatch> matches, bool use_ratio_test = true);
        
    private:
        
        Ptr<FeatureDetector> detector;
        Ptr<DescriptorExtractor> extractor;
        Ptr<DescriptorMatcher> matcher;
    
    };
}