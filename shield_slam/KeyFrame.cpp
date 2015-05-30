#include "KeyFrame.hpp"

using namespace cv;
using namespace std;

namespace vslam {
    
    KeyFrame::KeyFrame(void)
    {
        R = Mat::eye(3, 3, CV_64F);
        t = Mat::zeros(3, 1, CV_64F);
        
        local_map.clear();
        orb_kp.clear();
        orb_desc = Mat();
        
        insertion_frame_count = 0;
    }
    
    KeyFrame::KeyFrame(Mat &rot_mat, Mat &trans_mat, vector<MapPoint> &map,
                       KeypointArray &total_kp, Mat &total_desc)
    {
        R = rot_mat.clone();
        t = trans_mat.clone();
        
        local_map = map;
        
        orb_kp = total_kp;
        orb_desc = total_desc.clone();
        
        insertion_frame_count = 0;
    }
    
    vector<Point3f> KeyFrame::Get3DPoints(void)
    {
        vector<Point3f> point_3D;
        for (int i=0; i<local_map.size(); i++)
        {
            point_3D.push_back(local_map.at(i).GetPoint3D());
        }
        
        return point_3D;
    }
    
    Mat KeyFrame::GetDescriptors(void)
    {
        Mat desc = Mat::zeros((int)local_map.size(), local_map.at(0).GetDesc().cols, CV_8U);
        
        for (int i=0; i<local_map.size(); i++)
        {
            local_map.at(i).GetDesc().copyTo(desc.row(i));
        }
        
        return desc;
    }
    
    void KeyFrame::GetKpDesc(PointArray &kp, Mat &desc)
    {
        desc = Mat::zeros((int)local_map.size(), local_map.at(0).GetDesc().cols, CV_8U);
        kp.clear();
        
        for (int i=0; i<local_map.size(); i++)
        {            
            kp.push_back(local_map.at(i).GetPoint2D());
            local_map.at(i).GetDesc().copyTo(desc.row(i));
        }
    }
    
    float KeyFrame::ComputeMedianDepth(void)
    {
        Mat R_t = R.row(2);
        R_t = R_t.t();
        
        float z_world = t.at<float>(2);
        
        vector<float> depths;
        
        for (int i=0; i<local_map.size(); i++)
        {
            Mat point_3D = Mat(local_map.at(i).GetPoint3D());
            point_3D.convertTo(point_3D, CV_64F);
            
            float z = R_t.dot(point_3D) + z_world;
            
            depths.push_back(z);
        }
        
        sort(depths.begin(), depths.end());
        
        return depths[(depths.size()-1)/2];
    }
    
    KeypointArray KeyFrame::GetTrackedKeypoints(void)
    {
        KeypointArray kp_array;
        
        for (int i=0; i<local_map.size(); i++)
        {
            MapPoint mp = local_map.at(i);
            KeyPoint kp = KeyPoint(mp.GetPoint2D().x, mp.GetPoint2D().y, 1);
            kp_array.push_back(kp);
        }
        
        return kp_array;
    }
}
