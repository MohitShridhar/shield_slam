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
        
        KeyFrame GetCurrKeyFrame(void)
        {
            if (!keyframes.empty())
                return keyframes.back();
            else
            {
                KeyFrame empty_kf;
                return empty_kf;
            }
        }
        
        vector<KeyFrame> GetKeyFrames(void) { return keyframes; }
        
        Ptr<ORB> orb_handler;
        
    private:
        
        Initializer initializer;
        
        void LoadIntrinsicParameters(void);
        void AppendCameraPose(Mat rot, Mat pos);
        void CommpoundCameraPose();
    
    protected:
        
        Mat initial_frame;
        vector<KeyFrame> keyframes;
        vector<Mat> world_camera_pos, world_camera_rot;
        
        State curr_state, prev_state;
    };
    
}

#endif /* defined(__shield_slam__VSlam__) */
