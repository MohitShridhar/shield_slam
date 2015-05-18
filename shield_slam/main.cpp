#include "VSlam.hpp"
#include <opencv2/opencv.hpp>

#include "UpdateListener.hpp"
#include "Visualizer.hpp"

using namespace cv;
using namespace std;

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
    init_imgs.push_back(frame.clone());
    
    for (int i=0; i<frame_increments; i++)
    {
        cap >> frame;
    }
    
    init_imgs.push_back(frame.clone());
    slam.Initialize(init_imgs);
    
    vector<Point3d> pc;
    pc.push_back(Point3d(298.28506, 156.14775, 55.964255));
    pc.push_back(Point3d(315.02551, 108.96662, 59.215987));
    pc.push_back(Point3d(320.2616, 111.32796, 59.528822));
    pc.push_back(Point3d(249.23425, 144.78455, 61.087847));
    pc.push_back(Point3d(229.66425, 114.83986, 67.154276));
    pc.push_back(Point3d(265.14923, 109.7708	, 0.66122794));
    pc.push_back(Point3d(310.97034, 97.487946, 59.802634));
    pc.push_back(Point3d(311.36179, 98.14399	, 0.60576868));
    pc.push_back(Point3d(317.31879, 119.97291, 55.282712));
    pc.push_back(Point3d(249.69025, 130.36581, 64.855546));
    pc.push_back(Point3d(290.26477, 96.127983, 63.655245));
    pc.push_back(Point3d(287.88705, 96.81031	, 0.62858152));
    pc.push_back(Point3d(283.5148, 132.31555, 63.004088));
    pc.push_back(Point3d(318.53055, 98.165489, 63.326758));
    pc.push_back(Point3d(317.51944, 98.334625, 62.627727));
    pc.push_back(Point3d(324.93204, 118.21966, 57.106465));
    pc.push_back(Point3d(310.72775, 97.6306 	, 0.63805091));
    pc.push_back(Point3d(255.27625, 149.54517, 62.568671));
    pc.push_back(Point3d(331.47989, 105.58896, 58.982801));
    pc.push_back(Point3d(233.92317, 128.26674, 66.456449));
    pc.push_back(Point3d(279.4086, 85.01812	, 0.65898931));
    pc.push_back(Point3d(312.88428, 102.09042, 60.403025));
    pc.push_back(Point3d(301.91449, 93.988525, 61.666197));
    pc.push_back(Point3d(290.88586, 94.94632	, 0.63291776));
    pc.push_back(Point3d(330.90057, 105.36166, 58.921653));
    pc.push_back(Point3d(320.95248, 123.92258, 56.426924));
    pc.push_back(Point3d(285.27722, 91.599052, 62.561494));
    pc.push_back(Point3d(283.51483, 132.31557, 63.004088));
    pc.push_back(Point3d(287.02347, 96.936592, 62.614912));
    pc.push_back(Point3d(316.21164, 97.363998, 62.890756));
    pc.push_back(Point3d(324.16336, 116.84482, 56.279033));
    pc.push_back(Point3d(317.87234, 96.625969, 61.460853));
    pc.push_back(Point3d(293.60657, 98.877327, 62.417048));
    pc.push_back(Point3d(313.39734, 96.997246, 60.31574));
    pc.push_back(Point3d(314.34769, 99.161987, 61.205393));
    pc.push_back(Point3d(315.51688, 97.776016, 61.721444));
    pc.push_back(Point3d(316.68546, 121.3801	, 0.55818069));
    pc.push_back(Point3d(245.7487, 121.89881, 68.264419));
    pc.push_back(Point3d(317.15231, 96.304588, 61.350095));
    pc.push_back(Point3d(323.97  , 103.28177, 57.835895));
    pc.push_back(Point3d(290.06076, 131.11992, 62.363333));
    pc.push_back(Point3d(319.98456, 118.15871, 56.977975));
    pc.push_back(Point3d(309.77563, 95.524971, 59.756756));
    pc.push_back(Point3d(298.01825, 142.38553, 61.41237));
    pc.push_back(Point3d(326.03836, 103.44984, 57.466501));
    pc.push_back(Point3d(327.70193, 117.98354, 56.893355));
    pc.push_back(Point3d(298.90778, 156.71869, 55.981082));
    pc.push_back(Point3d(326.37341, 123.87517, 56.890273));
    pc.push_back(Point3d(292.65497, 132.73708, 62.960064));
    pc.push_back(Point3d(317.3197, 117.17495, 56.503481));
    pc.push_back(Point3d(296.638  , 101.776 	, 0.62652326));
    pc.push_back(Point3d(248.41208, 129.22701, 64.465958));
    pc.push_back(Point3d(299.08054, 100.06966, 63.632655));
    pc.push_back(Point3d(326.49731, 104.21238, 57.43078));
    pc.push_back(Point3d(327.4068, 104.41611, 58.660525));
    pc.push_back(Point3d(322.49573, 98.023834, 60.99062));
    pc.push_back(Point3d(329.72705, 104.70547, 58.034146));
    pc.push_back(Point3d(270.78152, 77.182571, 65.293294));
    pc.push_back(Point3d(292.2836, 77.773712, 64.658713));
    pc.push_back(Point3d(249.5945, 128.83116, 64.71476));
    pc.push_back(Point3d(294.37582, 127.65857, 62.816519));
    pc.push_back(Point3d(312.04672, 119.59086, 54.922414));
    pc.push_back(Point3d(310.14929, 93.429413, 60.069096));
    pc.push_back(Point3d(279.20251, 128.0835	, 0.61765158));
    pc.push_back(Point3d(304.26126, 159.66014, 55.792022));
    pc.push_back(Point3d(288.03418, 95.175812, 61.192513));
    pc.push_back(Point3d(232.27769, 121.83206, 65.27878));
    pc.push_back(Point3d(289.67343, 76.829948, 64.317352));
    pc.push_back(Point3d(317.29749, 98.258919, 64.728874));
    pc.push_back(Point3d(251.04187, 129.57808, 65.090007));
    pc.push_back(Point3d(308.0694, 96.188499, 60.393912));
    pc.push_back(Point3d(249.8223, 134.65245, 65.194416));
    pc.push_back(Point3d(273.07867, 108.90443, 67.328447));
    pc.push_back(Point3d(289.81229, 76.781952, 64.277142));
    pc.push_back(Point3d(285.41632, 124.42144, 61.273432));
    pc.push_back(Point3d(280.16467, 128.03108, 62.137598));
    pc.push_back(Point3d(298.45801, 156.74184, 55.840462));
    pc.push_back(Point3d(305.48917, 93.663864, 62.231594));
    pc.push_back(Point3d(268.99841, 110.91107, 65.853852));
    
    
    UpdateCloud(pc);
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