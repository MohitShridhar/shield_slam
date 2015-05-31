#include "Tracking.hpp"

using namespace cv;
using namespace std;

namespace vslam {
    
    Ptr<ORB> Tracking::orb_handler;
    double Tracking::init_scale =  1.0f;
    bool Tracking::has_scale_init = false;
    
    bool Tracking::TrackMap(const cv::Mat &gray_frame, KeyFrame& kf, Mat &R, Mat &t, bool& new_kf_added)
    {
        Mat Rvec, tvec, pnp_inliers;

        Rodrigues(R, Rvec);
        tvec = t;

        // Find matches with reference to the keyframe
        Mat tar_desc;
        Mat tar_img = gray_frame;
        KeypointArray tar_kp;
        orb_handler->ExtractFeatures(tar_img, tar_kp, tar_desc);
        
        Mat debug_kp;
        drawKeypoints(gray_frame, tar_kp, debug_kp, Scalar(0, 0, 255));
        imshow("Frame KPs", debug_kp);
        
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
        
        double min_val, max_val;
        minMaxIdx(image_points, &min_val, &max_val);
        
        solvePnPRansac(object_points, image_points, camera_matrix, dist_coeff, Rvec, tvec,
                       true, 100, 0.006f * max_val, 0.24f * (double)(image_points.size()), pnp_inliers, CV_ITERATIVE);
        
        Rodrigues(Rvec, R);
        t = tvec;
        
        /*
        // Correct scale using current KF as reference:
        double curr_scale = FindLinearScale(R, t, image_points, object_points);
        if (!has_scale_init)
        {
            Tracking::init_scale = curr_scale;
            has_scale_init = true;
        }
        else
        {
            double scale_ratio = Tracking::init_scale / curr_scale;
            t *= scale_ratio;
            
            cout << scale_ratio << endl;
        }
        */
        
        kf.IncrementFrameCount();
        KeypointArray ref_kp = kf.GetTotalKeypoints();
        
        new_kf_added = false;
        if (NeedsNewKeyframe(kf, (int)ref_points.size(), (int)tar_kp.size(), (int)matches.size()))
        {
            Mat R_prev = kf.GetRotation();
            Mat t_prev = kf.GetTranslation();
            
            new_kf_added = NewKeyFrame(kf, R_prev, R, t_prev, t, ref_kp, tar_kp, ref_desc,
                                       tar_desc, matches, pnp_inliers, max_val, object_points);
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
                               Mat& ref_desc, Mat& tar_desc, vector<DMatch>& matches_2D_3D,
                               Mat& pnp_inliers, double max_val, vector<Point3f>& prev_pc)
    {
        vector<MapPoint> local_map;
        
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
        
        // TODO: check for still camera (corrupts scale)
        
        // Determine ratio factor for scale consistency check:
        const float ratio_factor = 1.5f * ORB_SCALE_FACTOR;
        
        // Build a hash map of existing point cloud points
        map<int, Point3f> existing_pc;
        for (int i=0; i<pnp_inliers.size().height; i++)
        {
            existing_pc[pnp_inliers.at<int>(i)] = prev_pc.at(i);
        }
        
        /*
        // Find fundamental matrix to determine outliers:
        PointArray ref_points, tar_points;
        for (int i=0; i<full_orb_matches.size(); i++)
        {
            ref_points.push_back(kp1[full_orb_matches[i].queryIdx].pt);
            tar_points.push_back(kp2[full_orb_matches[i].trainIdx].pt);
        }
        
        vector<uchar> f_status(full_orb_matches.size());
        findFundamentalMat(ref_points, tar_points, f_status, FM_RANSAC, 0.006 * max_val, 0.99);
        */
         
        int num_good_points = 0;
        for (int i=0; i<full_orb_matches.size(); i++)
        {
            /*
            if (!f_status[i])
                continue;
            */
             
            Point3f point_3D;
            
            KeyPoint ref_kp = kp1[full_orb_matches[i].queryIdx];
            KeyPoint tar_kp = kp2[full_orb_matches[i].trainIdx];
            
            // Check if the point already exists in the map
            if (existing_pc.count(full_orb_matches[i].queryIdx))
            {
                point_3D = existing_pc[full_orb_matches[i].queryIdx];
            }
            else
            {
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
                
                point_3D = Point3f(ref_point_3D.at<double>(0), ref_point_3D.at<double>(1), ref_point_3D.at<double>(2));
            }
            
            Mat desc;
            tar_desc.row(full_orb_matches[i].trainIdx).copyTo(desc);
            
            // Create MapPoint:
            MapPoint mp;
            mp.SetPoint3D(point_3D);
            mp.SetPoint2D(tar_kp.pt);
            mp.SetDesc(desc);
            local_map.push_back(mp);
            
            num_good_points++;
        }
        
        if (num_good_points >= TRIANGULATION_MIN_POINTS)
        {
            kf = KeyFrame(R2, t2, local_map, kp2, tar_desc);
            return true;
        }
        
        return false;
    }
    
    void Tracking::Normalize3DPoints(vector<Point3f> &input_points, vector<Point3f> &norm_points)
    {
        float max_x = numeric_limits<float>::min(), min_x = numeric_limits<float>::max();
        float max_y = numeric_limits<float>::min(), min_y = numeric_limits<float>::max();
        float max_z = numeric_limits<float>::min(), min_z = numeric_limits<float>::max();
        
        for (int i=0; i<input_points.size(); i++)
        {
            float x = input_points.at(i).x;
            float y = input_points.at(i).y;
            float z = input_points.at(i).z;
            
            if (x > max_x)
                max_x = x;
            if (x < min_x)
                min_x = x;
            
            if (y > max_y)
                max_y = y;
            if (y < min_y)
                min_y = y;
            
            if (z > max_z)
                max_z = z;
            if (z < min_z)
                min_z = z;
        }
        
        float x_range = fabs(max_x - min_x);
        float y_range = fabs(max_y - min_y);
        float z_range = fabs(max_z - min_z);
        
        norm_points.clear();
        
        for (int i=0; i<input_points.size(); i++)
        {
            Point3f norm_point;
            
            norm_point.x = input_points.at(i).x / x_range;
            norm_point.y = input_points.at(i).y / y_range;
            norm_point.z = input_points.at(i).z / z_range;
            
            norm_points.push_back(norm_point);
        }
    }
    
    // Reference: http://dare.uva.nl/document/2/113942
    double Tracking::FindLinearScale(Mat &R, Mat &t, vector<Point2f> &image_points, vector<Point3f> &object_points)
    {
        double scale = 1.0f;
        
        Matx34d Pcam(R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0),
                  R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1),
                  R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2));
        
        Mat mat_2D = Mat(2, (int)image_points.size(), CV_64F);
        Mat mat_3D = Mat(3, (int)image_points.size(), CV_64F);
        
        for (int i=0; i<image_points.size(); i++)
        {
            mat_2D.at<double>(0, i) = image_points.at(i).x;
            mat_2D.at<double>(1, i) = image_points.at(i).y;
            
            mat_3D.at<double>(0, i) = object_points.at(i).x;
            mat_3D.at<double>(1, i) = object_points.at(i).y;
            mat_3D.at<double>(2, i) = object_points.at(i).z;
        }
        
        Mat mat_2D_homogeneous;
        vconcat(mat_2D, Mat::ones(1, mat_2D.size().width, mat_2D.type()), mat_2D_homogeneous);
        
        // Convert to homogeneous coordinates (u, v, 1):
        Mat Qw = camera_matrix.inv() * mat_2D_homogeneous;
        
        // Build matrix A and vector b
        cv::Mat_<double> A ( 2 * mat_2D.size().width, 1 );
        cv::Mat_<double> b ( 2 * mat_2D.size().width, 1 );
        
        cv::Matx13d r1( R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2) );
        cv::Matx13d r2( R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2) );
        cv::Matx13d r3( R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2) );
        
        cv::Matx23d r12;
        cv::vconcat(r1,r2,r12);
        
        cv::Matx21d tu (Pcam(0,3), Pcam(1,3));
        
        
        cv::Matx21d temp1, temp2;
        for ( int i = 0; i < mat_3D.size().width; i++ ) {
            //temp1 = ( Pcam(1:2,1:3) * X3D(1:3,i)  -
            //         (Pcam(3,1:3) * X3D(1:3,i)) * Qw(1:2,i));
            cv::Matx31d pointX ( mat_3D.at<double>(0,i),
                                mat_3D.at<double>(1,i),
                                mat_3D.at<double>(2,i) );
            cv::Matx21d pointx ( Qw.at<double>(0,i),
                                Qw.at<double>(1,i) );
            
            cv::subtract( ((cv::Mat)r12) * (cv::Mat)pointX,
                         (cv::Mat)pointx * ((cv::Mat)((cv::Mat)r3 * (cv::Mat)pointX)),
                         temp1);
            
            //temp2 = Pcam(3,4) * Qw(1:2,i) - Pcam(1:2,4);
            cv::subtract( Pcam(2,3) * (cv::Mat)pointx, (cv::Mat)tu, temp2);
            
            A.at<double>(i*2,   0) = temp2(0);
            A.at<double>(i*2+1, 0) = temp2(1);
            b.at<double>(i*2,   0) = temp1(0);
            b.at<double>(i*2+1, 0) = temp1(1);
        }
        
        cv::Mat scalemat = ((A.t() * b) / (A.t() * A));
        scale = scalemat.at<double>(0, 0);
        
        
        // METHOD 4
        double scale2;
        cv::Mat_<double> A2 ( mat_2D.size().width, 1 );
        cv::Mat_<double> b2 ( mat_2D.size().width, 1 );
        
        for ( int i = 0; i < mat_3D.size().width; i++ ) {
            cv::Matx31d pointX ( mat_3D.at<double>(0,i),
                                mat_3D.at<double>(1,i),
                                mat_3D.at<double>(2,i) );
            cv::Point2d pointx ( Qw.at<double>(0,i),
                                Qw.at<double>(1,i) );
            
            //                  Pcam(2,4) * ( Qw(1,i)            / Qw(2,i))            - Pcam(1,4);
            A2.at<double>(i, 0) = Pcam(1,3) * (pointx.x / pointx.y) - Pcam(0,3);
            
            //                  (Pcam(1,1:3)-Pcam(2,1:3) *
            //                  (Qw(1,i) / Qw(2,i))) * X3D(1:3,i)
            cv::Mat temp = ((cv::Mat) r1 - ((cv::Mat) r2 * (pointx.x / pointx.y))) * (cv::Mat) pointX;
            b2.at<double>(i, 0) = temp.at<double>(0,0);
        }
        scale2 = (double)((cv::Mat)((cv::Mat(A.t() * A)) * A.t() *b)).at<double>(0,0);
        
        scalemat = ((A2.t() * b2) / (A2.t() * A2));
        scale2 = scalemat.at<double>(0, 0);

        return scale2;
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