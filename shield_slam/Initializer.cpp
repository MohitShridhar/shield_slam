#include "Initializer.hpp"

using namespace cv;
using namespace std;

namespace vslam
{
    
    Initializer::Initializer()
    {
        orb_handler = new ORB(500, false);
    }
    
    void Initializer::BaseLineTriangulation(vector<cv::Mat> &init_imgs)
    {
        Mat img_ref, img_tar;
        
        // TODO: this is assuming init_imgs has only two images. Check for better initialization
        img_ref = init_imgs.at(0);
        img_tar = init_imgs.at(1);
        
        
        KeypointArray keypoints_ref_, keypoints_tar_;
        Mat desc_ref_, desc_tar_;
        vector<DMatch> matches;
        
        orb_handler->ExtractFeatures(img_ref, keypoints_ref_, desc_ref_);
        orb_handler->ExtractFeatures(img_tar, keypoints_tar_, desc_tar_);
        
        orb_handler->MatchFeatures(desc_ref_, desc_tar_, matches);
        
        cout << matches.size() << endl;
        
        // Debug:
//        Mat debug_draw_img;
//        drawMatches(img_ref, keypoints_ref_, img_tar, keypoints_tar_, matches, debug_draw_img);
//        
//        imshow("orb matches", debug_draw_img);
//        waitKey(0);
        //------
        
        
        
    }
    
}
