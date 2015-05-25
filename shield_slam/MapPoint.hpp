#ifndef __shield_slam__MapPoint__
#define __shield_slam__MapPoint__

#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>


using namespace cv;
using namespace std;

namespace vslam {
    
    class MapPoint
    {
        
    public:
        void SetPoint3D(Point3f coord) { point_3D = coord; }
        Point3f GetPoint3D(void) { return point_3D; }
        
        void SetPoint2D(Point2f coord) { point_2D = coord; }
        Point2f GetPoint2D(void) { return point_2D; }
        
        void SetDesc(Mat& desc) { descriptor = desc.clone(); }
        Mat GetDesc(void) { return descriptor; }
        
    private:
        
    protected:
        Point2f point_2D;
        Point3f point_3D;
        Mat descriptor;
        
    };
}

#endif /* defined(__shield_slam__MapPoint__) */