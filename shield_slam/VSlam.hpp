#ifndef __shield_slam__VSlam__
#define __shield_slam__VSlam__

#include <opencv2/opencv.hpp>

#include "Initializer.hpp"
#include "Common.hpp"

using namespace cv;
using namespace std;

namespace vslam
{
    
    class VSlam
    {
    public:
        
        VSlam();
        virtual ~VSlam() = default;
        
        void Initialize(vector<Mat>& init_imgs);
        void ProcessFrame(Mat& img);
        
    private:
        
        Initializer initializer;
        
        void LoadIntrinsicParameters(void);
        
    };
    
}

#endif /* defined(__shield_slam__VSlam__) */
