#include "Tracking.hpp"

using namespace cv;
using namespace std;

namespace vslam {
    
    void Tracking::PosePnP(Ptr<ORB> orb_handler, const cv::Mat &gray_frame, KeyFrame& kf, Mat &R, Mat &t)
    {
        Mat Rvec, tvec, pnp_inliers;
        
        Rodrigues(R, Rvec);
        tvec = t;

        // Find matches with reference to the keyframe
        Mat tar_desc;
        Mat tar_img = gray_frame;
        KeypointArray tar_kp;
        orb_handler->ExtractFeatures(tar_img, tar_kp, tar_desc);
        
        Mat ref_desc;
        PointArray ref_points;
        vector<Point3f> ref_point_cloud;
        
        kf.GetKpDesc(ref_points, ref_desc);
        ref_point_cloud = kf.Get3DPoints();
        
        vector<DMatch> matches;
        orb_handler->MatchFeatures(ref_desc, tar_desc, matches, true);
        
        // Prepare image and object points:
        vector<Point2f> image_points;
        vector<Point3f> object_points;
        
        for (int i=0; i<matches.size(); i++)
        {
            Point2f image_point = tar_kp[matches[i].trainIdx].pt;
            image_points.push_back(image_point);
            
            Point3f object_point = ref_point_cloud[matches[i].queryIdx];
            object_points.push_back(object_point);
        }
        
        solvePnPRansac(object_points, image_points, camera_matrix, dist_coeff, Rvec, tvec,
                       true, 100, 8.0, 100, pnp_inliers);
        
        Rodrigues(Rvec, R);
        t = tvec;
    }
    
    void Tracking::Triangulate(const KeyPoint &ref_keypoint, const KeyPoint &tar_keypoint, const Mat &P1, const Mat &P2, Mat &point_3D)
    {
        Point3d ref_point (ref_keypoint.pt.x, ref_keypoint.pt.y, 1.0);
        Point3d tar_point (tar_keypoint.pt.x, tar_keypoint.pt.y, 1.0);
        
        Matx31d out_3D = IterativeLinearLSTriangulation(ref_point, tar_point, P1, P2);
        point_3D = Mat(out_3D);
    }
    
    // Reference: https://perception.inrialpes.fr/Publications/1997/HS97/HartleySturm-cviu97.pdf & Mastering Practical OpenCV
    Mat_<double> Tracking::LinearLSTriangulation(const Point3d &u1, const Point3d &u2, const Mat &P1, const Mat &P2)
    {
        Matx43d A(
                  u1.x * P1.at<double>(2,0) - P1.at<double>(0,0), u1.x * P1.at<double>(2,1) - P1.at<double>(0,1), u1.x * P1.at<double>(2,2) - P1.at<double>(0,2),
                  u1.y * P1.at<double>(2,0) - P1.at<double>(1,0), u1.y * P1.at<double>(2,1) - P1.at<double>(1,1), u1.y * P1.at<double>(2,2) - P1.at<double>(1,2),
                  u2.x * P2.at<double>(2,0) - P2.at<double>(0,0), u2.x * P2.at<double>(2,1) - P2.at<double>(0,1), u2.x * P2.at<double>(2,2) - P2.at<double>(0,2),
                  u2.y * P2.at<double>(2,0) - P2.at<double>(1,0), u2.y * P2.at<double>(2,1) - P2.at<double>(1,1), u2.y * P2.at<double>(2,2) - P2.at<double>(1,2)
                  );
        Matx41d B(
                  -( u1.x * P1.at<double>(2,3) - P1.at<double>(0,3) ),
                  -( u1.y * P1.at<double>(2,3) - P1.at<double>(1,3) ),
                  -( u2.x * P2.at<double>(2,3) - P2.at<double>(0,3) ),
                  -( u2.y * P2.at<double>(2,3) - P2.at<double>(1,3) )
                  );
        
        Mat_<double> X;
        solve( A, B, X, DECOMP_SVD );
        
        return X;
    }
    
    Matx31d Tracking::IterativeLinearLSTriangulation(const Point3d &u1, const Point3d &u2, const Mat &P1, const Mat &P2)
    {
        double wi1 = 1;
        double wi2 = 1;
        
        Matx41d X;
        
        for ( int i = 0; i < TRIANGULATION_LS_ITERATIONS; i++ ) {
            Mat_<double> X_ = LinearLSTriangulation( u1, u2, P1, P2 );
            X = Matx41d( X_(0), X_(1), X_(2), 1.0 );
            
            // Recalculate weights
            double p2x1 = Mat_<double>( P1.row( 2 ) * Mat(X) ).at<double>(0);
            double p2x2 = Mat_<double>( P2.row( 2 ) * Mat(X) ).at<double>(0);
            
            // Breaking point
            if ( fabs( wi1 - p2x1 ) <= TRIANGULATION_LS_EPSILON && fabs( wi2 - p2x2 ) <= TRIANGULATION_LS_EPSILON )
                break;
            
            wi1 = p2x1;
            wi2 = p2x2;
            
            // Reweight equations and solve
            Matx43d A(
                      ( u1.x * P1.at<double>(2,0) - P1.at<double>(0,0) ) / wi1, ( u1.x * P1.at<double>(2,1) - P1.at<double>(0,1) ) / wi1, ( u1.x * P1.at<double>(2,2) - P1.at<double>(0,2) ) / wi1,
                      ( u1.y * P1.at<double>(2,0) - P1.at<double>(1,0) ) / wi1, ( u1.y * P1.at<double>(2,1) - P1.at<double>(1,1) ) / wi1, ( u1.y * P1.at<double>(2,2) - P1.at<double>(1,2) ) / wi1,
                      ( u2.x * P2.at<double>(2,0) - P2.at<double>(0,0) ) / wi2, ( u2.x * P2.at<double>(2,1) - P2.at<double>(0,1) ) / wi2, ( u2.x * P2.at<double>(2,2) - P2.at<double>(0,2) ) / wi2,
                      ( u2.y * P2.at<double>(2,0) - P2.at<double>(1,0) ) / wi2, ( u2.y * P2.at<double>(2,1) - P2.at<double>(1,1) ) / wi2, ( u2.y * P2.at<double>(2,2) - P2.at<double>(1,2) ) / wi2
                      );
            Matx41d B(
                      -( u1.x * P1.at<double>(2,3) - P1.at<double>(0,3) ) / wi1,
                      -( u1.y * P1.at<double>(2,3) - P1.at<double>(1,3) ) / wi1,
                      -( u2.x * P2.at<double>(2,3) - P2.at<double>(0,3) ) / wi2,
                      -( u2.y * P2.at<double>(2,3) - P2.at<double>(1,3) ) / wi2
                      );
            
            solve( A, B, X_, DECOMP_SVD );
            X = Matx41d( X_(0), X_(1), X_(2), 1.0 );
        }
        
        return Matx31d( X(0), X(1), X(2) );
        
    }

}