#include "VSlam.hpp"
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

namespace vslam
{
    
    VSlam::VSlam()
    {
        
    }
    
}

int main(int argc, char** argv)
{
    VideoCapture cap("/Users/MohitSridhar/NCSV/Stanford/CS231M/projects/project2/Project2/Project2/test-videos/mona-lisa-blur.avi");
    
    if (!cap.isOpened())
    {
        cout << "failed to open video file" << endl;
        return -1;
    }
    
    
    Mat frame;
    cap >> frame;
    
    for ( ; ; )
    {
        cap >> frame;
        if (frame.empty())
            break;
        
        if (waitKey(30) == 27)
        {
            break;
        }
    }
    
    waitKey(0);
    
    return 0;
}