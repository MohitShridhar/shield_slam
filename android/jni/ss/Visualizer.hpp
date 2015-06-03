#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <pcl/common/common.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/ros/conversions.h>

#include <boost/thread.hpp>

#include <Eigen/Eigen>

#include "../ss/Common.hpp"

#define CAMERA_POSE_SCALE 0.5

#define X_SCALE 1.0
#define Y_SCALE 1.0
#define Z_SCALE 560.0

#define MAX_PC_VAL 100

using namespace cv;
using namespace std;

void InitializeVisualizer();
void RunVisualizationOnly();

void WaitForVisualizationThread();
void RunVisualizationThread();

void UpdateCloud(const vector<Point3d>& point_cloud, const int r, const int g, const int b, bool clear);
void AddCamera(const Mat& R, const Mat& t);
