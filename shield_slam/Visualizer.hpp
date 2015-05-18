#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <pcl/common/common.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <boost/thread.hpp>

using namespace cv;
using namespace std;

void InitializeVisualizer();
void RunVisualizationOnly();

void WaitForVisualizationThread();
void RunVisualizationThread();

void UpdateCloud(const vector<Point3d>& point_cloud);