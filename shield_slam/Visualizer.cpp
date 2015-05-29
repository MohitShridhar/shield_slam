#include "Visualizer.hpp"

using namespace cv;
using namespace std;
using namespace Eigen;

static bool update_pc = false, update_camera = false;
static boost::mutex update_pc_model_mutex, add_camera_mutex;

// Reference: Mastering Practical OpenCV

boost::thread* viz_th_ = NULL;
boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer ( new pcl::visualization::PCLVisualizer("Shield SLAM", true));

pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_ptr (new pcl::PointCloud<pcl::PointXYZRGB>);

std::deque<std::pair<std::string,pcl::PolygonMesh> > cam_meshes;
int ipolygon[18] = {0,1,2,  0,3,1,  0,4,3,  0,2,4,  3,1,4,   2,4,1};
int camera_count = 0;

bool aggregate_view_frustrums = false;

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
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud");
    viewer->addCoordinateSystem (1);
    viewer->setRepresentationToWireframeForAllActors();
    viewer->initCameraParameters ();
}

void RunVisualizationOnly()
{
    if (!viewer->wasStopped())
    {
        viewer->spinOnce(100);
        
        boost::mutex::scoped_lock update_pc_lock(update_pc_model_mutex);
        if (update_pc)
        {
            if (!viewer->updatePointCloud(cloud_ptr, "cloud"))
            {
                viewer->addPointCloud(cloud_ptr, "cloud");
            }
            
            update_pc = false;
        }
        update_pc_lock.unlock();
        
        boost::mutex::scoped_lock update_camera_lock(add_camera_mutex);
        if (update_camera)
        {
            
            if (aggregate_view_frustrums)
            {
                camera_count = 0;
                for (int i=0; i<cam_meshes.size(); i++)
                {
                    viewer->removePolygonMesh(to_string(camera_count));
                    viewer->addPolygonMesh(cam_meshes.at(i).second, to_string(camera_count));
                    camera_count++;
                }
            }
            else
            {
                viewer->removePolygonMesh(to_string(0));
                viewer->addPolygonMesh(cam_meshes.back().second, to_string(0));
            }
    
            update_camera = false;
        }
        
        update_camera_lock.unlock();
        
        viewer->setRepresentationToWireframeForAllActors();
    }
}

void UpdateCloud(const vector<Point3d>& point_cloud, const int r, const int g, const int b, bool clear)
{
    boost::mutex::scoped_lock updateLock(update_pc_model_mutex);
    update_pc = true;
    
    if (clear)
        cloud_ptr->clear();
    
    for (int i=0; i<point_cloud.size(); i++)
    {
        Vec3b rgbv(r, g, b);
        
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
        
        point.x = point_cloud[i].x * X_SCALE;
        point.y = point_cloud[i].y * Y_SCALE;
        point.z = point_cloud[i].z * Z_SCALE;
        
        uint32_t rgb = ((uint32_t)rgbv[2] << 16 | (uint32_t)rgbv[1] << 8 | (uint32_t)rgbv[0]);
        point.rgb = *reinterpret_cast<float*>(&rgb);
        
        cloud_ptr->push_back(point);
    }
    
    cloud_ptr->width = (uint32_t) cloud_ptr->points.size();
    cloud_ptr->height = 1;
    
    updateLock.unlock();
}

inline pcl::PointXYZRGB Eigen2PointXYZRGB(Eigen::Vector3f v, Eigen::Vector3f rgb) {
    pcl::PointXYZRGB p(rgb[0],rgb[1],rgb[2]); p.x = v[0]; p.y = v[1]; p.z = v[2];
    return p;
}


void AddCamera(const Mat& R, const Mat& t)
{
    boost::mutex::scoped_lock updateLock(add_camera_mutex);
    update_camera = true;
    
    Matrix3f r_mat;
    r_mat << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2),
             R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2),
             R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2);
    
    Vector3f t_vec = Vector3f(t.at<double>(0) * X_SCALE, t.at<double>(1) * Y_SCALE, t.at<double>(2) * 1.0f);
    t_vec = -r_mat.transpose() * t_vec;
    
    t_vec = Vector3f(t_vec.x(), t_vec.y(), t_vec.z());
    
    Vector3f vec_right = r_mat.row(0).normalized() * CAMERA_POSE_SCALE;
    Vector3f vec_up = -r_mat.row(1).normalized() * CAMERA_POSE_SCALE;
    Vector3f vec_forward = r_mat.row(2).normalized() * CAMERA_POSE_SCALE;
    
    Vector3f rgb(255, 0, 0);
    
	pcl::PointCloud<pcl::PointXYZRGB> mesh_cld;
	mesh_cld.push_back(Eigen2PointXYZRGB(t_vec, rgb));
	mesh_cld.push_back(Eigen2PointXYZRGB(t_vec + vec_forward + vec_right/2.0 + vec_up/2.0,rgb));
	mesh_cld.push_back(Eigen2PointXYZRGB(t_vec + vec_forward + vec_right/2.0 - vec_up/2.0,rgb));
	mesh_cld.push_back(Eigen2PointXYZRGB(t_vec + vec_forward - vec_right/2.0 + vec_up/2.0,rgb));
	mesh_cld.push_back(Eigen2PointXYZRGB(t_vec + vec_forward - vec_right/2.0 - vec_up/2.0,rgb));
    
	pcl::PolygonMesh pm;
	pm.polygons.resize(6);
	for(int i=0;i<6;i++)
		for(int _v=0;_v<3;_v++)
			pm.polygons[i].vertices.push_back(ipolygon[i*3 + _v]);
    
    pcl::toROSMsg(mesh_cld,pm.cloud);
	cam_meshes.push_back(std::make_pair("camera" + std::to_string(camera_count),pm));
    
    updateLock.unlock();
}



