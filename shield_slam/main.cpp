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
	void update(std::vector<cv::Point3d> pcld,
				std::vector<cv::Matx34d> cameras) {
		UpdateCloud(pcld);
		
		vector<cv::Matx34d> v = cameras;
		for(unsigned int i=0;i<v.size();i++) {
			stringstream ss; ss << "camera" << i;
			cv::Matx33f R;
			R(0,0)=v[i](0,0); R(0,1)=v[i](0,1); R(0,2)=v[i](0,2);
			R(1,0)=v[i](1,0); R(1,1)=v[i](1,1); R(1,2)=v[i](1,2);
			R(2,0)=v[i](2,0); R(2,1)=v[i](2,1); R(2,2)=v[i](2,2);
//			visualizerShowCamera(R,cv::Vec3f(v[i](0,3),v[i](1,3),v[i](2,3)),255,0,0,0.2,ss.str());
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
    
    int frame_increments = 20;
    
    vslam::VSlam slam = vslam::VSlam();
    
    // Initialize:
    Mat frame;
    cap >> frame;
    
    vector<Mat> init_imgs;
    init_imgs.resize(2);
    init_imgs.at(0) = frame.clone();
    
    for ( ; ; )
    {
        for (int i=0; i<frame_increments; i++)
            cap >> frame;
        
        
        init_imgs.at(1) = frame.clone();
        
        slam.Initialize(init_imgs);
        if (slam.getCurrState() == vslam::VSlam::TRACKING)
            break;

    }
    
    // Load Initialized Map:
    vector<MapPoint> global_map = slam.GetGlobalMap();
    vector<Point3d> init_pc;
    for (int i=0; i<global_map.size(); i++)
    {
        init_pc.push_back(global_map.at(i).GetPos());
    }
    
    UpdateCloud(init_pc);
    RunVisualizationOnly();
    
    for ( ; ; )
    {
        for (int i=0; i<frame_increments; i++)
        {
            cap >> frame;
        }
        
        cap >> frame;
        if (frame.empty())
            break;
        
//        imshow("Input", frame);
//        waitKey(0);
        
        if (waitKey(30) == 27)
        {
            break;
        }
        
        RunVisualizationOnly();
    }
    
    waitKey(0);

//    WaitForVisualizationThread();
    return 0;
}