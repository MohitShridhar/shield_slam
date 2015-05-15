#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "Common.hpp"

using namespace cv;
using namespace std;

namespace vslam {
    
    class ORB
    {
        
    public:
        
        ORB(int n_features, bool use_gpu);
        virtual ~ORB() = default;
        
        void ExtractFeatures (Mat& img, PointArray& img_keypoints, Mat& img_desc);
        void MatchFeatures (PointArray& features_ref, PointArray& features_tar, vector<DMatch> matches);
        
    private:
        
        Ptr<FeatureDetector> dector;
        Ptr<DescriptorExtractor> extractor;
        Ptr<DescriptorMatcher> matcher;
    
    };
}