#ifndef __shield_slam__Optimizer__
#define __shield_slam__Optimizer__

#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include "../ss/Common.hpp"
#include "../ss/KeyFrame.hpp"
#include "../ss/MapPoint.hpp"

using namespace cv;
using namespace std;

namespace vslam {
    
    class Optimizer
    {
    public:
        static void BundleAdjust(vector<KeyFrame>& keyframes);
        
    private:

        
    protected:
        
        
        
    };
}

#endif /* defined(__shield_slam__Optimizer__) */