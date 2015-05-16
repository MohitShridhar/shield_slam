#include "VSlam.hpp"

using namespace cv;
using namespace std;

namespace vslam
{
    
    VSlam::VSlam()
    {
        LoadIntrinsicParameters();
    }
    
    void VSlam::Initialize(vector<cv::Mat> &init_imgs)
    {
        initializer.InitializeMap(init_imgs);
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
    }
    
}
