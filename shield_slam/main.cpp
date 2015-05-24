#include "VSlam.hpp"
#include <opencv2/opencv.hpp>

#include "UpdateListener.hpp"
#include "Visualizer.hpp"
#include "MapPoint.hpp"

using namespace cv;
using namespace std;
using namespace vslam;

class VisualizerListener : public UpdateListener {
    
public:
	void update(vector<MapPoint> global_map, vector<Mat> camera_rot, vector<Mat> camera_pos) {
        vector<Point3d> init_pc;
        for (int i=0; i<global_map.size(); i++)
        {
            init_pc.push_back(global_map.at(i).GetPos());
        }
        UpdateCloud(init_pc);
		
		for(unsigned int i=0;i<camera_rot.size();i++) {
            AddCamera(camera_rot.at(i), camera_pos.at(i));
		}
	}
};

int main(int argc, char** argv)
{
    VideoCapture cap("/Users/MohitSridhar/NCSV/Stanford/CS231M/projects/shield_slam/indoor.avi");
    
    if (!cap.isOpened())
    {
        cout << "failed to open video file" << endl;
        return -1;
    }

    Ptr<VisualizerListener> visualizerListener = new VisualizerListener;
    InitializeVisualizer();
    
    vslam::VSlam slam = vslam::VSlam();
    
    // Initialize:
    Mat frame;
    cap >> frame;
    
    // Load Initialized Map:
    RunVisualizationOnly();
    
    slam.ProcessFrame(frame);
    
    for (int i=0; i<1; i++)
    {
        cap >> frame;
    }
    
    for ( ; ; )
    {
        cap >> frame;
        if (frame.empty())
            break;
        
        slam.ProcessFrame(frame);
        
//        imshow("Input", frame);
//        waitKey(0);
        
        if (waitKey(30) == 27)
        {
            break;
        }
        
        visualizerListener->update(slam.GetGlobalMap(), slam.GetCameraRot(), slam.GetCameraPose());
        RunVisualizationOnly();
        waitKey(0);
    }
    
    waitKey(0);

//    WaitForVisualizationThread();
    return 0;
}