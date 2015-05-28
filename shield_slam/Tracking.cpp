#include "Tracking.hpp"

using namespace cv;
using namespace std;

namespace vslam {
    
    Ptr<ORB> Tracking::orb_handler;
    
    bool Tracking::TrackMap(const cv::Mat &gray_frame, KeyFrame& kf, Mat &R, Mat &t, bool& new_kf_added)
    {
        Mat Rvec, tvec, pnp_inliers;
        
        Mat R_prev = R.clone();
        Mat t_prev = t.clone();
        
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
                       true, 100, 8.0, 100, pnp_inliers, CV_ITERATIVE);
        
        Rodrigues(Rvec, R);
        t = tvec;
        
        kf.IncrementFrameCount();
        KeypointArray ref_kp = kf.GetTotalKeypoints();
        
        new_kf_added = false;
        if (NeedsNewKeyframe(kf, (int)ref_points.size(), (int)tar_kp.size(), (int)matches.size()))
        {
            new_kf_added = NewKeyFrame(kf, R_prev, R, t_prev, t, ref_kp, tar_kp, ref_desc, tar_desc, matches);
            return new_kf_added;
        }
        
        return true;
    }
    
    bool Tracking::NeedsNewKeyframe(KeyFrame& kf, int num_kf_kp, int num_tar_kp, int num_kf_matches)
    {
//        cout << num_tar_kp << " " << (1.0 * num_kf_matches) / num_kf_kp << " " << kf.GetFrameCountSinceInsertion() << endl;
        
        if (num_tar_kp < KEYFRAME_MIN_KEYPOINTS)
            return true;
        
        if ((1.0 * num_kf_matches) / num_kf_kp < KEYFRAME_MIN_MATCH_RATIO)
            return true;
        
        if (kf.GetFrameCountSinceInsertion() > KEYFRAME_MAX_FRAME_COUNT_SINCE_INSERTION)
            return true;
        
        return false;
    }
    
    bool Tracking::NewKeyFrame(KeyFrame &kf, Mat &R1, Mat &R2, Mat &t1, Mat &t2,
                               KeypointArray &kp1, KeypointArray &kp2,
                               Mat& ref_desc, Mat& tar_desc, vector<DMatch>& matches_2D_3D)
    {
        vector<MapPoint> local_map;
        
        /*
        // Add existing triangulated 3D points
        vector<Point3f> prev_kf_local_map = kf.Get3DPoints();
        vector<MapPoint> existing_points;
        for (int i=0; i<matches_2D_3D.size(); i++)
        {
            MapPoint mp;
            mp.SetPoint2D(kp2[matches_2D_3D[i].trainIdx].pt);
            mp.SetPoint3D(prev_kf_local_map.at(matches_2D_3D[i].queryIdx));
            
            Mat desc;
            tar_desc.row(matches_2D_3D[i].trainIdx).copyTo(desc);
            mp.SetDesc(desc);
            
            existing_points.push_back(mp);
        }
        local_map.insert(local_map.end(), existing_points.begin(), existing_points.end());
        */
        
        // Do full feature matching:
        vector<DMatch> full_orb_matches;
        ref_desc = kf.GetTotalDescriptors();
        orb_handler->MatchFeatures(ref_desc, tar_desc, full_orb_matches);
        
        // Intrinsic parameters (3D->2D) for projection error checking:
        float cam_fx = camera_matrix.at<double>(0, 0);
        float cam_fy = camera_matrix.at<double>(1, 1);
        float cam_cx = camera_matrix.at<double>(0, 2);
        float cam_cy = camera_matrix.at<double>(1, 2);
        
        // P1 = K[R1|t1]
        Mat ref_origin = -R1.t()*t1;
        Mat P1 = (Mat_<double>(3, 4) << R1.at<double>(0, 0), R1.at<double>(0, 1), R1.at<double>(0, 2), t1.at<double>(0),
                                        R1.at<double>(1, 0), R1.at<double>(1, 1), R1.at<double>(1, 2), t1.at<double>(1),
                                        R1.at<double>(2, 0), R1.at<double>(2, 1), R1.at<double>(2, 2), t1.at<double>(2));
        P1 = camera_matrix * P1;
        
        // P2 = K[R2|t2]
        Mat tar_origin = -R2.t()*t2;
        Mat P2 = (Mat_<double>(3, 4) << R2.at<double>(0, 0), R2.at<double>(0, 1), R2.at<double>(0, 2), t2.at<double>(0),
                                        R2.at<double>(1, 0), R2.at<double>(1, 1), R2.at<double>(1, 2), t2.at<double>(1),
                                        R2.at<double>(2, 0), R2.at<double>(2, 1), R2.at<double>(2, 2), t2.at<double>(2));
        P2 = camera_matrix * P2;
        
        const float ratio_factor = 1.5f * ORB_SCALE_FACTOR;
    
        int num_good_points = 0;
        for (int i=0; i<full_orb_matches.size(); i++)
        {
            KeyPoint ref_kp = kp1[full_orb_matches[i].queryIdx];
            KeyPoint tar_kp = kp2[full_orb_matches[i].trainIdx];
            
            float ref_scale_factor = pow(ORB_SCALE_FACTOR, ref_kp.octave);
            float tar_scale_factor = pow(ORB_SCALE_FACTOR, tar_kp.octave);
            
            Mat ref_point_3D = Mat(3, 1, CV_64F, Scalar(0));
            Triangulate(ref_kp, tar_kp, P1, P2, ref_point_3D);
            
            Mat ref_point_3D_tp = ref_point_3D.t();
            
            // Check that the point is finite:
            if (!isfinite(ref_point_3D.at<double>(0)) ||
                !isfinite(ref_point_3D.at<double>(1)) ||
                !isfinite(ref_point_3D.at<double>(2)))
            {
                continue;
            }
            
            // Check parallax:
            Mat ref_normal = ref_point_3D - ref_origin;
            float ref_dist = norm(ref_normal);
            
            Mat tar_normal = ref_point_3D - tar_origin;
            float tar_dist = norm(tar_normal);
            
            if (ref_dist == 0.0 || tar_dist == 0.0)
                continue;
            
            // OLD REPROJECTION ERROR CHECKING:
            /*
            float cos_parallax = ref_normal.dot(tar_normal) / (ref_dist * tar_dist);
            
            // Check that the point is in front of the reference camera:
            if (ref_point_3D.at<double>(2) <= 0.0 && cos_parallax < 0.9998)
            {
                continue;
            }
            
            Mat tar_point_3D = R2 * ref_point_3D + t2;
            
            // Check that the point is in front of the target camera:
            if (tar_point_3D.at<double>(2) <= 0.0 && cos_parallax < 0.9998)
            {
                continue;
            }

            // Check reprojection error for reference image:
            float ref_reproj_x, ref_reproj_y;
            ref_reproj_x = cam_fx * (ref_point_3D.at<double>(0) / ref_point_3D.at<double>(2)) + cam_cx;
            ref_reproj_y = cam_fy * (ref_point_3D.at<double>(1) / ref_point_3D.at<double>(2)) + cam_cy;
            
            float ref_square_error = (ref_reproj_x - ref_kp.pt.x) * (ref_reproj_x - ref_kp.pt.x) +
            (ref_reproj_y - ref_kp.pt.y) * (ref_reproj_y - ref_kp.pt.y);
            
            if (ref_square_error > REPROJECTION_ERROR_TH)
            {
                continue;
            }
            
            // Check reprojection error for target image:
            float tar_reproj_x, tar_reproj_y;
            tar_reproj_x = cam_fx * (tar_point_3D.at<double>(0) / tar_point_3D.at<double>(2)) + cam_cx;
            tar_reproj_y = cam_fy * (tar_point_3D.at<double>(1) / tar_point_3D.at<double>(2)) + cam_cy;
            
            float tar_square_error = (tar_reproj_x - tar_kp.pt.x) * (tar_reproj_x - tar_kp.pt.x) +
            (tar_reproj_y - tar_kp.pt.y) * (tar_reproj_y - tar_kp.pt.y);
            
            if (tar_square_error > REPROJECTION_ERROR_TH)
            {
                continue;
            }
            */
            
            // Check if point is in front of the cameras:
            float z1 = R1.row(2).dot(ref_point_3D_tp)+t1.at<double>(2);
            if (z1 <= 0)
                continue;
            
            float z2 = R2.row(2).dot(ref_point_3D_tp)+t2.at<double>(2);
            if (z2 <= 0)
                continue;
            
            // Check reprojection error for reference camera:
            float x1 = R1.row(0).dot(ref_point_3D_tp)+t1.at<double>(0);
            float y1 = R1.row(1).dot(ref_point_3D_tp)+t1.at<double>(1);
            float inv_z1 = 1.0 / z1;
            
            float u1 = cam_fx * x1 * inv_z1 + cam_cx;
            float v1 = cam_fy * y1 * inv_z1 + cam_cy;
            
            float err_x1 = u1 - ref_kp.pt.x;
            float err_y1 = v1 - ref_kp.pt.y;
            if (err_x1 * err_x1 + err_y1 * err_y1 > REPROJECTION_ERROR_CHI * ref_scale_factor * ref_scale_factor)
                continue;
        
            // Check reprojection error for target camera:
            float x2 = R2.row(0).dot(ref_point_3D_tp)+t2.at<double>(0);
            float y2 = R2.row(1).dot(ref_point_3D_tp)+t2.at<double>(1);
            float inv_z2 = 1.0 / z2;
            
            float u2 = cam_fx * x2 * inv_z2 + cam_cx;
            float v2 = cam_fy * y2 * inv_z2 + cam_cy;
            
            float err_x2 = u2 - tar_kp.pt.x;
            float err_y2 = v2 - tar_kp.pt.y;
            if (err_x2 * err_x2 + err_y2 * err_y2 > REPROJECTION_ERROR_CHI * tar_scale_factor * tar_scale_factor)
                continue;
             
            // Check scale consistency:
            float ratio_dist = ref_dist / tar_dist;
            float ratio_octave = ref_scale_factor / tar_scale_factor;
            
            if (ratio_dist * ratio_factor < ratio_octave || ratio_dist > ratio_octave * ratio_factor)
                continue;
                
            // Create MapPoint:
            Mat desc;
            tar_desc.row(full_orb_matches[i].trainIdx).copyTo(desc);
            
            Point3f point_3D = Point3f(ref_point_3D.at<double>(0), ref_point_3D.at<double>(1), ref_point_3D.at<double>(2));
            
            MapPoint mp;
            mp.SetPoint3D(point_3D);
            mp.SetPoint2D(tar_kp.pt);
            mp.SetDesc(desc);
            local_map.push_back(mp);
            
            num_good_points++;
        }
        
        if (num_good_points >= TRIANGULATION_MIN_POINTS)
        {
            Mat new_R = R1;
            Mat new_t = t1;
            
            kf = KeyFrame(new_R, new_t, local_map, kp2, tar_desc);
            return true;
        }
        
        return false;
    }
    
    void Tracking::FilterPnPInliers(vector<Point3f> &object_points, vector<Point2f> &image_points, Mat& inliers)
    {
        vector<Point3f> filtered_obj_points;
        vector<Point2f> filtered_img_points;
        
        int inlier_idx = 0;
        for (int i=0; i<object_points.size(); i++)
        {
            if (inliers.at<int>(inlier_idx) == i)
            {
                filtered_obj_points.push_back(object_points.at(i));
                filtered_img_points.push_back(image_points.at(i));
                
                inlier_idx++;
            }
        }
        
        object_points = filtered_obj_points;
        image_points = filtered_img_points;
    }
    
    void Tracking::Triangulate(const KeyPoint &ref_keypoint, const KeyPoint &tar_keypoint, const Mat &P1, const Mat &P2, Mat &point_3D)
    {
        /*
        cv::Mat A(4,4,CV_64F);
        
        A.row(0) = ref_keypoint.pt.x*P1.row(2)-P1.row(0);
        A.row(1) = ref_keypoint.pt.y*P1.row(2)-P1.row(1);
        A.row(2) = tar_keypoint.pt.x*P2.row(2)-P2.row(0);
        A.row(3) = tar_keypoint.pt.y*P2.row(2)-P2.row(1);
        
        cv::Mat u,w,vt;
        cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);
        point_3D = vt.row(3).t();
        point_3D = point_3D.rowRange(0,3) / point_3D.at<float>(3);
        */
        
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