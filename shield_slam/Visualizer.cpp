#include "Visualizer.hpp"

using namespace cv;
using namespace std;

static bool update;
static boost::mutex update_model_mutex;

boost::thread* viz_th_ = NULL;
boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer ( new pcl::visualization::PCLVisualizer("Shield SLAM", true));

pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_ptr (new pcl::PointCloud<pcl::PointXYZRGB>);

void RunVisualizationThread()
{
    viz_th_ = new boost::thread(RunVisualizationOnly);
}

void WaitForVisualizationThread()
{
    viz_th_->join();
}

void InitializeVisualizer()
{
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud");
    viewer->addCoordinateSystem (1);
    viewer->initCameraParameters ();
}

void RunVisualizationOnly()
{
    if (!viewer->wasStopped())
    {
        viewer->spinOnce(100);
        
        boost::mutex::scoped_lock updateLock(update_model_mutex);
        
        if (update)
        {
            if (!viewer->updatePointCloud(cloud_ptr, "cloud"))
            {
//                viewer->removePointCloud("Initial");
                viewer->addPointCloud(cloud_ptr, "cloud");
            }
            
            update = false;
        }
        updateLock.unlock();
    }
}

void UpdateCloud(const vector<Point3d>& point_cloud)
{
    boost::mutex::scoped_lock updateLock(update_model_mutex);
    update = true;
    
    cloud_ptr->clear();
    
    for (int i=0; i<point_cloud.size(); i++)
    {
        Vec3b rgbv(0, 255, 0);
        
        // Check for invalid points:
        if (point_cloud[i].x != point_cloud[i].x ||
            point_cloud[i].y != point_cloud[i].y ||
            point_cloud[i].z != point_cloud[i].z ||
            isnan(point_cloud[i].x) ||
            isnan(point_cloud[i].y) ||
            isnan(point_cloud[i].z))
        {
            continue;
        }
        
        pcl::PointXYZRGB point;
        
        point.x = point_cloud[i].x;
        point.y = point_cloud[i].y;
        point.z = point_cloud[i].z;
        
        uint32_t rgb = ((uint32_t)rgbv[2] << 16 | (uint32_t)rgbv[1] << 8 | (uint32_t)rgbv[0]);
        point.rgb = *reinterpret_cast<float*>(&rgb);
        
        cloud_ptr->push_back(point);
    }
    
    cloud_ptr->width = (uint32_t) cloud_ptr->points.size();
    cloud_ptr->height = 1;
    
    updateLock.unlock();
}

