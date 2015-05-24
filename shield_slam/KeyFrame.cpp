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

}