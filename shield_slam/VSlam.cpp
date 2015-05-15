#include "VSlam.hpp"

using namespace cv;
using namespace std;

namespace vslam
{
    
    VSlam::VSlam()
    {
        
    }
    
    void VSlam::Initialize(vector<cv::Mat> &init_imgs)
    {
        initializer.BaseLineTriangulation(init_imgs);
    }
    
}
