#ifndef __shield_slam__VSlam__
#define __shield_slam__VSlam__

#include <opencv2/opencv.hpp>

#include "Initializer.hpp"
#include "MapPoint.hpp"
#include "Common.hpp"
#include "KeyFrame.hpp"
#include "Tracking.hpp"
#include "ORB.hpp"

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
            LOST = 3,
        };
        
        State GetCurrState(void) { return curr_state; }
        vector<Mat> GetCameraPose(void) { return world_camera_pos; }
        vector<Mat> GetCameraRot(void) { return world_camera_rot; }
        vector<MapPoint> GetGlobalMap(void) { return global_map_; }
        KeyFrame GetCurrKeyFrame(void) { return curr_kf; }
        
        Ptr<ORB> orb_handler;
        
    private:
        
        Initializer initializer;
        
        void LoadIntrinsicParameters(void);
        void CompoundCameraPose(Mat rot, Mat pos);
        void AppendCameraPose(Mat rot, Mat pos);
    
    protected:
        
        Mat initial_frame;
        KeyFrame curr_kf;
        vector<Mat> world_camera_pos, world_camera_rot;
        
        vector<MapPoint> global_map_;
        
        State curr_state, prev_state;
    };
    
}

#endif /* defined(__shield_slam__VSlam__) */
