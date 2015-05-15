#ifndef __shield_slam__Initializer__
#define __shield_slam__Initializer__

#include <opencv2/opencv.hpp>

#include "ORB.hpp"

using namespace cv;
using namespace std;

namespace vslam
{
    
    class Initializer
    {
    public:
        
        Initializer();
        virtual ~Initializer() = default;
        
        void BaseLineTriangulation(vector<Mat>& init_imgs);
        
    private:
        
        Ptr<ORB> orb_handler;
        
        
    };
    
}

#endif /* defined(__shield_slam__Initializer__) */
