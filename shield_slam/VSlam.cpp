#include "VSlam.hpp"

using namespace cv;
using namespace std;

namespace vslam
{
    
    VSlam::VSlam()
    {
        curr_state = NOT_INITIALIZED;
    }
    
    void VSlam::Initialize(vector<cv::Mat> &init_imgs)
    {
        if (curr_state == NOT_INITIALIZED)
        {
            LoadIntrinsicParameters();
            curr_state = INITIALIZING;
        }
        
        if (curr_state == INITIALIZING)
        {
            if(initializer.InitializeMap(init_imgs, global_map_)) // FIX ME: global should be updated
            {
                curr_state = TRACKING;
            }
        }
    }
    
    void VSlam::LoadIntrinsicParameters()
    {
        FileStorage fs("/Users/MohitSridhar/NCSV/Stanford/CS231M/projects/shield_slam/shield_slam/CameraIntrinsics.yaml", FileStorage::READ);
        
        if (!fs.isOpened())
        {
            CV_Error(0, "VSlam: Could not load calibration file");
        }
        
        fs["cameraMatrix"] >> camera_matrix;
        fs["distCoeffs"] >> dist_coeff;
        fs["imageSize"] >> img_size;
    }
    
}
