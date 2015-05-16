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
        
        
        vector<DMatch> matches;
        PointArray ref_matches, tar_matches;
        
        orb_handler->DetectAndMatch(img_ref, img_tar, matches, ref_matches, tar_matches);
        
        
        
        
    }
    
}
