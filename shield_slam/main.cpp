#include "VSlam.hpp"
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    VideoCapture cap("/Users/MohitSridhar/NCSV/Stanford/CS231M/projects/shield_slam/indoor.avi");
    
    if (!cap.isOpened())
    {
        cout << "failed to open video file" << endl;
        return -1;
    }
    
    int frame_increments = 20;
    
    vslam::VSlam slam;
    
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
    }
    
    waitKey(0);
    
    return 0;
}