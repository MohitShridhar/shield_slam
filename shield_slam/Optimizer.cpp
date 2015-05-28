//#include "Optimizer.hpp"
//
//using namespace cv;
//using namespace g2o;
//
//namespace vslam
//{
//    // Reference: ORB SLAM - Optimizer
//    int Optimizer::EstimatePose(KeyFrame& kf)
//    {
//        vector<MapPoint> local_map = kf.GetMap();
//        
//        SparseOptimizer optimizer;
//        BlockSolverX::LinearSolverType *linear_solver;
//        
//        linear_solver = new LinearSolverDense<BlockSolverX::PoseMatrixType>();
//        BlockSolverX *solver_ptr = new BlockSolverX(linear_solver);
//        
//        OptimizationAlgorithmLevenberg *solver = new OptimizationAlgorithmLevenberg(solver_ptr);
//        optimizer.setAlgorithm(solver);
//        
//        optimizer.setVerbose(false);
//        
//        int num_init_points = 0;
//        
//        // Set camera vertex:
//        Mat R = kf.GetRotation();
//        Mat t = kf.GetTranslation();
//        Mat pose = (Mat_<double>(3, 4) << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0),
//                                          R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1),
//                                          R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2));
//        
//        VertexSE3Expmap * v_SE3 = new VertexSE3Expmap();
//        v_SE3->setEstimate(toSE3Quat(pose));
//        v_SE3->setId(0);
//        v_SE3->setFixed(false);
//        optimizer.addVertex(v_SE3);
//        
//        vector<EdgeSE3Expmap*> edges;
//        vector<VertexSBAPointXYZ*> vertices;
//        vector<float> inv_sigmas2;
//        vector<size_t> index_edge;
//        
//        const int N = (int)local_map.size();
//        edges.reserve(N);
//        vertices.reserve(N);
//        inv_sigmas2.reserve(N);
//        index_edge.reserve(N);
//        
//        const float delta = sqrt(5.991);
//        
//        for (int i=0; i<local_map.size(); i++)
//        {
//            // Set vertex point:
//            MapPoint mp = local_map.at(i);
//            
//            VertexSBAPointXYZ *point = new VertexSBAPointXYZ();
//            point->setEstimate(toVector3d(mp.GetPoint3D()));
//            point->setId(i+1);
//            point->setFixed(true);
//            
//            optimizer.addVertex(point);
//            vertices.push_back(point);
//            
//            num_init_points++;
//            
//            // Set edge:
//            Eigen::Matrix<double, 2, 1> obs;
//            Point2f kp = mp.GetPoint2D();
//            obs << kp.x, kp.y;
//            
//            EdgeSE3Expmap *edge = new EdgeSE3Expmap();
//            
//            edge->setVertex(0, dynamic_cast<OptimizableGraph::Vertex*>(optimizer.vertex(i+1)));
//            edge->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
////            edge->setMeasurement(obs);
//        }
//        
//        return 0;
//    }
//    
//    SE3Quat Optimizer::toSE3Quat(const cv::Mat &cvT)
//    {
//        Eigen::Matrix<double,3,3> R;
//        R << cvT.at<float>(0,0), cvT.at<float>(0,1), cvT.at<float>(0,2),
//        cvT.at<float>(1,0), cvT.at<float>(1,1), cvT.at<float>(1,2),
//        cvT.at<float>(2,0), cvT.at<float>(2,1), cvT.at<float>(2,2);
//        
//        Eigen::Matrix<double,3,1> t(cvT.at<float>(0,3), cvT.at<float>(1,3), cvT.at<float>(2,3));
//        
//        return SE3Quat(R,t);
//    }
//    
//    Eigen::Matrix<double,3,1> Optimizer::toVector3d(const Point3f &pos)
//    {
//        Eigen::Matrix<double,3,1> v;
//        v << pos.x, pos.y, pos.z;
//        
//        return v;
//    }
//    
//}
