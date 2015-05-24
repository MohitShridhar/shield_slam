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
        void SetPoint(Point3f coord) { point = coord; }
        Point3f GetPos(void) { return point; }
        
        void SetDesc(Mat& desc) { descriptor = desc.clone(); }
        Mat GetDesc(void) { return descriptor; }
        
    private:
        
    protected:
        Point3f point;
        Mat descriptor;
        
    };
}

#endif /* defined(__shield_slam__MapPoint__) */