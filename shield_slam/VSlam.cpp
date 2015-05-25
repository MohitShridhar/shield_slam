#include "VSlam.hpp"

using namespace cv;
using namespace std;

namespace vslam
{
    
    VSlam::VSlam()
    {
        LoadIntrinsicParameters();
        
        Mat init_camera_rot = Mat::eye(3, 3, CV_64F);
        Mat init_camera_pos = Mat::zeros(3, 1, CV_64F);
        world_camera_rot.push_back(init_camera_rot);
        world_camera_pos.push_back(init_camera_pos);
        
        orb_handler = new ORB(500, true);
        
        curr_state = NOT_INITIALIZED;
    }
    
    void VSlam::ProcessFrame(cv::Mat &img)
    {
        Mat frame;
        cvtColor(img, frame, CV_RGB2GRAY);
        
        if (curr_state == NOT_INITIALIZED)
        {
            initial_frame = frame.clone();
            curr_state = INITIALIZING;
        }
        
        if (curr_state == INITIALIZING)
        {
            if(initializer.InitializeMap(orb_handler, initial_frame, frame, curr_kf, global_map_))
            {
                AppendCameraPose(curr_kf.GetRotation(), curr_kf.GetTranslation());
                curr_state = TRACKING;
            }
        }
        
        if (curr_state == TRACKING)
        {
            Mat R_vec = world_camera_rot.back().clone();
            Mat t_vec = world_camera_pos.back().clone();
            
            Tracking::PosePnP(orb_handler, frame, curr_kf, R_vec, t_vec);
            
            // TODO: check if lost
            AppendCameraPose(R_vec, t_vec);
        }
    }
    
    void VSlam::AppendCameraPose(Mat rot, Mat pos)
    {
        world_camera_rot.push_back(world_camera_rot.back() * rot);
        world_camera_pos.push_back(world_camera_pos.back() + pos);
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
