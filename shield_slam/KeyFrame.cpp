#include <opencv2/features2d/features2d.hpp>

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
            mp.SetPoint(points.at(i));
            mp.SetDesc(descriptors.at(i));
            
            local_map.push_back(mp);
        }
        
    }
    
    KeyFrame::KeyFrame(Mat &rot_mat, Mat &trans_mat, vector<MapPoint> &map)
    {
        R = rot_mat.clone();
        t = trans_mat.clone();
        
        local_map = map;
    }
    
    vector<Point3f> KeyFrame::Get3DPoints(void)
    {
        vector<Point3f> point_3D;
        for (int i=0; i<local_map.size(); i++)
        {
            point_3D.push_back(local_map.at(0).GetPos());
        }
        
        return point_3D;
    }
    
    Mat KeyFrame::GetDescriptors(void)
    {
        Mat desc = Mat::zeros((int)local_map.size(), local_map.at(0).GetDesc().cols, CV_64F);
        
        for (int i=0; i<local_map.size(); i++)
        {
            desc.row(i) = local_map.at(i).GetDesc();
        }
        
        return desc;
    }

}