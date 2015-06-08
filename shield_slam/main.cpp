#include "VSlam.hpp"

#include <opencv2/opencv.hpp>
#include <ctime>
#include <iostream>

#include "Augmentor.hpp"
#include "MapPoint.hpp"
#include "UpdateListener.hpp"
#include "Visualizer.hpp"

using namespace cv;
using namespace std;
using namespace vslam;

class VisualizerListener : public UpdateListener {
    
public:
	void update(vector<KeyFrame> keyframes, Mat camera_rot, Mat camera_pos) {
        
        vector<Point3d> init_pc;
        vector<Point3d> kf_pc;
        
        if (keyframes.empty())
            return;
        
        vector<MapPoint> local_map = keyframes.back().GetMap();
        vector<MapPoint> global_map;
        
        for (int i=0; i<local_map.size(); i++)
        {
            kf_pc.push_back(local_map.at(i).GetPoint3D());
        }
        
        for (int i=0; i<keyframes.size(); i++)
        {
            vector<MapPoint> kf_map = keyframes.at(i).GetMap();
            
            for (int j=0; j<kf_map.size(); j++)
            {
                init_pc.push_back(kf_map.at(j).GetPoint3D());
            }
        }
        
        
// GLOBAL MAP:
       UpdateCloud(init_pc, 0, 255, 0, true);
        
// LOCAL MAP:
//       UpdateCloud(kf_pc, 0, 0, 255, true); 
        
// GLOBAL MAP & LOCAL MAP:
//      UpdateCloud(init_pc, 0, 255, 0, true);
//		UpdateCloud(kf_pc, 255, 255, 255, false);
        
        AddCamera(camera_rot, camera_pos);
	}
};

#define KAI_PATH "/Users/neo/Dropbox/231m/shield_slam/"
#define MOHIT_PATH "/Users/MohitSridhar/NCSV/Stanford/CS231M/projects/shield_slam/"

int main(int argc, char** argv)
{
    VideoCapture cap(string(KAI_PATH).append("shield_vid.mp4"));
//    VideoCapture cap(string(KAI_PATH).append("indoor.mov"));
//    VideoCapture cap(string(MOHIT_PATH).append("indoor.avi"));
//    VideoCapture cap("/Users/MohitSridhar/Downloads/kitti_youtube.avi");
//    VideoCapture cap("/Users/MohitSridhar/Downloads/VID_20150530_120719.mp4");
    
    if (!cap.isOpened())
    {
        cout << "failed to open video file" << endl;
        return -1;
    }

    // Initialize visualizer and load initial map
    Ptr<VisualizerListener> visualizerListener = new VisualizerListener;
    InitializeVisualizer();
    RunVisualizationOnly();
    
    // Initialize SLAM
    Mat frame;
    Size size(640, 480);
    vslam::VSlam slam = vslam::VSlam();

    while (true) {
        cap >> frame;
        if (frame.empty()) {
            break;
        }
        
        resize(frame, frame, size);
        
        clock_t start = clock();
        slam.ProcessFrame(frame);
        clock_t end = clock();
        
        double processFrameDuration = (end - start) / (double) CLOCKS_PER_SEC;
        cout << "processFrameDuration: " << processFrameDuration << endl;
        
        if (waitKey(30) == 27) {
            break;
        }
        
        // Update visualizer
        visualizerListener->update(slam.GetKeyFrames(), slam.GetCameraRot().back(), slam.
                                   GetCameraPose().back());
        RunVisualizationOnly();
        
        // Draw translation and rotation information
        Augmentor augmentor;
        KeyFrame currKeyFrame = slam.GetCurrKeyFrame();
        Mat translationMatrix = currKeyFrame.GetTranslation();
        augmentor.DisplayTranslation(frame, translationMatrix);
        Mat rotationMatrix = currKeyFrame.GetRotation();
        augmentor.DisplayRotation(frame, rotationMatrix);
        
        // Draw keypoints
        KeypointArray keypoints = currKeyFrame.GetTrackedKeypoints();
        Mat trackedFeatures;
        Scalar kpColor = Scalar(255, 0, 0);
        drawKeypoints(frame, keypoints, trackedFeatures, kpColor);
        imshow("Tracked Features", trackedFeatures);
    }
    
    waitKey(0);

    WaitForVisualizationThread();
    return 0;
}




























