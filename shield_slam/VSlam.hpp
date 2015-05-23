#ifndef __shield_slam__VSlam__
#define __shield_slam__VSlam__

#include <opencv2/opencv.hpp>

#include "Initializer.hpp"
#include "MapPoint.hpp"
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
        
        enum State{
            NOT_INITIALIZED = 0,
            INITIALIZING = 1,
            TRACKING = 2,
        };
        
        State getCurrState(void) { return curr_state; }
        
        vector<MapPoint> GetGlobalMap(void) { return global_map_; }
        
    private:
        
        Initializer initializer;
        
        void LoadIntrinsicParameters(void);
    
    protected:
        
        vector<MapPoint> global_map_;
        
        State curr_state, prev_state;
        
    };
    
}

#endif /* defined(__shield_slam__VSlam__) */
