#include "KeyFrame.hpp"

using namespace cv;
using namespace std;

namespace vslam {
    
    KeyFrame::KeyFrame(void)
    {
        R = Mat::eye(3, 3, CV_64F);
        t = Mat::zeros(3, 1, CV_64F);
        
        local_map.clear();
    }
    
    /* Depreciated */
    KeyFrame::KeyFrame(Mat &rot_mat, Mat &trans_mat, vector<Point3f> &points, vector<Mat> &descriptors)
    {
        R = rot_mat.clone();
        t = trans_mat.clone();
        
        // Assumption: contains only triangulated point descriptors
        assert(points.size() == descriptors.size());
        
        local_map.clear();
        
        for (int i=0; i<points.size(); i++)
        {
            MapPoint mp;
            mp.SetPoint3D(points.at(i));
            mp.SetDesc(descriptors.at(i));
            
            local_map.push_back(mp);
        }
    }
    
    KeyFrame::KeyFrame(Mat &rot_mat, Mat &trans_mat, vector<MapPoint> &map, KeypointArray &total_kp, Mat &total_desc)
    {
        R = rot_mat.clone();
        t = trans_mat.clone();
        
        local_map = map;
        orb_kp = total_kp;
        orb_desc = total_desc.clone();
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
}
