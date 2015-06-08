#include "Initializer.hpp"

using namespace cv;
using namespace std;

namespace vslam
{
    Initializer::Initializer() {}
    
    bool Initializer::InitializeMap(Ptr<ORB> orb_handler, Mat &img_ref, Mat &img_tar, vector<KeyFrame> &keyframes)
    {
        // Match ORB Features:
        Mat matched_tar_desc;
        vector<DMatch> matches;
        PointArray ref_matches, tar_matches;
        KeypointArray ref_kp, tar_kp;
        Mat ref_desc, tar_desc;
        
        orb_handler->DetectAndMatch(img_ref, img_tar, matches, ref_matches, tar_matches, matched_tar_desc, ref_kp, tar_kp, ref_desc, tar_desc);
        
        // Undistort key points using camera intrinsics:
        PointArray undist_ref_matches, undist_tar_matches;
        
        /*
        undistortPoints(ref_matches, undist_ref_matches, camera_matrix, dist_coeff);
        undistortPoints(tar_matches, undist_tar_matches, camera_matrix, dist_coeff);
        */
        
        
        undist_ref_matches = ref_matches;
        undist_tar_matches = tar_matches;
        
         
        // Compute homography and fundamental matrices:
        Mat H = findHomography(undist_ref_matches, undist_tar_matches, CV_RANSAC, 3);
        Mat F = findFundamentalMat(undist_ref_matches, undist_tar_matches, CV_FM_RANSAC, 3, 0.99);
        
        // Decide between homography and fundamental matrix:
        vector<bool> h_inliers, f_inliers;
        int h_num_inliers = 0, f_num_inliers;
        
        float SH = CheckHomography(undist_ref_matches, undist_tar_matches, H, h_inliers, h_num_inliers);
        float SF = CheckFundamental(undist_ref_matches, undist_tar_matches, F, f_inliers, f_num_inliers);
        
        
        /*
        float SH = 0.0f, SF = 0.0f;
        vector<bool> h_inliers, f_inliers;
        int h_num_inliers = 0, f_num_inliers;
        
        Mat H = FindHomography(undist_ref_matches, undist_tar_matches, SH, h_inliers, h_num_inliers);
        Mat F = FindFundamental(undist_ref_matches, undist_tar_matches, SF, f_inliers, f_num_inliers);
        */
         
        float RH = SH / (SH + SF);
        
        PointArray ref_inliers, tar_inliers;
        Mat P1 = Mat::eye(3, 4, CV_64F);
        Mat P2 = Mat::eye(3, 4, CV_64F);
        
        // Clear states:
        R = Mat();
        t = Mat();
        point_cloud_3D.clear();
        triangulated_state.clear();
        bool success = false;
        
        // Estimate camera pose based on the chosen model:
        if (RH > HOMOGRAPHY_SELECTION_THRESHOLD)
        {
            success = ReconstructHomography(undist_ref_matches, undist_tar_matches,
                                            matches, h_inliers, h_num_inliers,
                                            H, R, t, point_cloud_3D, triangulated_state);
        }
        else
        {
            success = ReconstructFundamental(undist_ref_matches, undist_tar_matches,
                                             matches, f_inliers, f_num_inliers,
                                             F, R, t, point_cloud_3D, triangulated_state);
        }
        
        if (success)
        {
            // (REFACTOR) Load details into the current keyframe:
            vector<Point2f> points_2D;
            vector<Point3f> points_3D;
            
            int pc_idx = 0;
            vector<MapPoint> local_map;
            for (int i=0; i<tar_matches.size(); i++)
            {
                if (triangulated_state.at(i))
                {
                    Mat desc = matched_tar_desc.row(i);
                    
                    MapPoint mp;
                    mp.SetPoint3D(point_cloud_3D.at(pc_idx));
                    mp.SetPoint2D(tar_matches.at(i));
                    mp.SetDesc(desc);
                    
                    points_3D.push_back(point_cloud_3D.at(pc_idx));
                    points_2D.push_back(tar_matches.at(i));
                    
                    local_map.push_back(mp);
                    pc_idx++;
                }
            }
            
            // Compute scale factor
//            double scale_factor = Tracking::FindLinearScale(R, t, points_2D, points_3D);
//            Tracking::SetInitScale(scale_factor);
            
            KeyFrame kf = KeyFrame(R, t, local_map, tar_kp, tar_desc);
            keyframes.push_back(kf);
            
            // Scale Translation mat
            /*
            float inv_median_depth = 1.0 / kf.ComputeMedianDepth();
            
            Mat scaled_t = inv_median_depth * t;
            kf.SetTranslation(scaled_t);
            
            // Scaled map points
            vector<MapPoint> scaled_local_map;
            for (int i=0; i<local_map.size(); i++)
            {
                MapPoint mp = local_map.at(i);
                
                Point3f point_3D = mp.GetPoint3D() * inv_median_depth;
                point_3D.z *= 1000;
                
                mp.SetPoint3D(point_3D);
                scaled_local_map.push_back(mp);
            }
            kf.SetLocalMap(scaled_local_map);
             
             */
        }
        
        return success;
    }
    
    
    // Reference: http://stackoverflow.com/questions/8927771/computing-camera-pose-with-homography-matrix-based-on-4-coplanar-points
    void Initializer::CameraPoseHomography(Mat &H, Mat &pose)
    {
        pose = Mat::eye(3, 4, CV_64F);
        
        double norm1 = (double)norm(H.col(0));
        double norm2 = (double)norm(H.col(1));
        double t_norm = (norm1 + norm2) / 2.0f;
        
        Mat p1 = H.col(0);
        Mat p2 = pose.col(0);
        
        normalize(p1, p2);
        
        p1 = H.col(1);
        p2 = pose.col(1);
        
        normalize(p1, p2);
        
        Mat p3 = p1.cross(p2);
        Mat c2 = pose.col(2);
        p3.copyTo(c2);
        
        pose.col(3) = H.col(2) / t_norm;
        
//        pose = camera_matrix * pose;
    }
    
    // Reference: http://subokita.com/2014/03/26/structure-from-motion-using-farnebacks-optical-flow-part-2/
    void Initializer::CameraPoseFundamental(Mat &F, Mat &pose)
    {
        pose = Mat::eye(3, 4, CV_64F);
        
        Mat essential = camera_matrix.t() * F * camera_matrix;
        
        SVD svd(essential);
        const Mat W = (Mat_<double>(3, 3) <<
                       0.0, -1.0, 0.0,
                       1.0,  0.0, 0.0,
                       0.0,  0.0, 1.0);
        
        const Mat W_inv = W.inv();
        
        Mat R1 = svd.u * W * svd.vt;
        Mat T1 = svd.u.col(2);
        
        Mat R2 = svd.u * W_inv * svd.vt;
        Mat T2 = -svd.u.col(2);
        
        pose = (Mat_<double>(3, 4) <<
                R1.at<double>(0, 0), R1.at<double>(0, 1), R1.at<double>(0, 2), T1.at<double>(0, 0),
                R1.at<double>(1, 0), R1.at<double>(1, 1), R1.at<double>(1, 2), T1.at<double>(1, 0),
                R1.at<double>(2, 0), R1.at<double>(2, 1), R1.at<double>(2, 2), T1.at<double>(2, 0));
    }
    
    Mat Initializer::FindHomography(PointArray &ref_keypoints, PointArray &tar_keypoints, float &score, vector<bool> &match_inliers, int &num_inliers)
    {
        Mat H = Mat::eye(3, 3, CV_64F);
        match_inliers = vector<bool>(ref_keypoints.size(), false);
        num_inliers = 0;
        
        Mat T1, T2;
        PointArray ref_norm_kp, tar_norm_kp;
        
        Normalize(ref_keypoints, ref_norm_kp, T1);
        Normalize(tar_keypoints, tar_norm_kp, T2);
        
        Mat H_norm = findHomography(ref_norm_kp, tar_norm_kp, CV_RANSAC, 3);
        H_norm.convertTo(H_norm, CV_64F);
        
        H = T2.inv() * H_norm * T1;
        
        score = CheckHomography(ref_keypoints, tar_keypoints, H, match_inliers, num_inliers);
        
        return H;
    }
    
    
    Mat Initializer::FindFundamental(PointArray &ref_keypoints, PointArray &tar_keypoints, float &score, vector<bool> &match_inliers, int &num_inliers)
    {
        Mat F = Mat::eye(3, 3, CV_64F);
        match_inliers = vector<bool>(ref_keypoints.size(), false);
        num_inliers = 0;
        
        Mat T1, T2;
        PointArray ref_norm_kp, tar_norm_kp;
        
        Normalize(ref_keypoints, ref_norm_kp, T1);
        Normalize(tar_keypoints, tar_norm_kp, T2);
        
        Mat F_norm = findFundamentalMat(ref_norm_kp, tar_norm_kp, CV_FM_RANSAC, 3, 0.99);
        F = T2.inv() * F_norm * T1;
        
        score = CheckFundamental(ref_keypoints, tar_keypoints, F, match_inliers, num_inliers);
        
        return F;
    }
    
    // Reference: https://hal.archives-ouvertes.fr/inria-00075698/document
    bool Initializer::ReconstructHomography(PointArray &ref_keypoints, PointArray &tar_keypoints, vector<DMatch> &matches, vector<bool> &inliers, int &num_inliers, Mat &H, Mat &R, Mat &t, vector<Point3f> &points, vector<bool> &triangulated_state)
    {
        // A = K^-1 * H * K
        Mat A = camera_matrix.inv() * H * camera_matrix;
        
        // Compute SVD
        Mat w, U, V_tp, V;
        SVD::compute(A, w, U, V_tp, SVD::FULL_UV);
        V = V_tp.t();
        
        float s = determinant(U) * determinant(V);
        
        float d1 = w.at<double>(0);
        float d2 = w.at<double>(1);
        float d3 = w.at<double>(2);
        
        if (d3 > d2 || d2 > d1)
        {
            return false;
        }
        
        // Prepare 8 possible rotation (homography 8DOF), translation and scale matrices:
        vector<Mat> p_R, p_t, p_n;
        
        // 4 possibilities: {e1, e3} : ( {1, 1}, {1, -1}, {-1, 1}, {-1, -1} )
        float sqrt_prod_x1 = sqrt((d1*d1 - d2*d2) / (d1*d1 - d3*d3));
        float sqrt_prod_x3 = sqrt((d2*d2 - d3*d3) / (d1*d1 - d3*d3));
        
        float x1[] = {sqrt_prod_x1, sqrt_prod_x1, -sqrt_prod_x1, -sqrt_prod_x1};
        float x3[] = {sqrt_prod_x3, -sqrt_prod_x3, sqrt_prod_x3, -sqrt_prod_x3};
        
        // Case: d' > 0
        float sqrt_prod_sin_theta = sqrt((d1*d1 - d2*d2) * (d2*d2 - d3*d3)) / ((d1+d3) * d2);
        
        float cos_theta = (d2*d2 + d1*d3) / ((d1+d3) * d2);
        float sin_theta[] = {sqrt_prod_sin_theta, -sqrt_prod_sin_theta, -sqrt_prod_sin_theta, sqrt_prod_sin_theta};
        
        for (int i=0; i<4; i++)
        {
            /* 
             R' = |cos(theta), 0, -sin(theta)|
                  |    0,      1,      0,    |
                  |sin(theta), 0,  cos(theta)|
            */
            
            Mat rotation_prime = Mat::eye(3, 3, CV_64F);
            rotation_prime.at<double>(0, 0) = cos_theta;
            rotation_prime.at<double>(0, 2) = -sin_theta[i];
            rotation_prime.at<double>(2, 0) = sin_theta[i];
            rotation_prime.at<double>(2, 2) = cos_theta;
            
            Mat rotation_mat = s * U * rotation_prime * V_tp;
            p_R.push_back(rotation_mat);
            
            /*
            t' =          |  x1 |
                 (d1 - d3)|   0 |
                          | -x3 |
            */
            
            Mat translation_prime = Mat::zeros(3, 1, CV_64F);
            translation_prime.at<double>(0) = x1[i];
            translation_prime.at<double>(2) = -x3[i];
            translation_prime *= (d1-d3);
            
            Mat translation_mat = U * translation_prime;
            p_t.push_back(translation_mat / norm(translation_mat));
            
            /*
            n' = | x1 |
                 | 0  |
                 |-x3 |
            */
            
            Mat scale_prime = Mat::zeros(3, 1, CV_64F);
            scale_prime.at<double>(0) = x1[i];
            scale_prime.at<double>(2) = x3[i];
            
            Mat scale_mat = V * scale_prime;
            if (scale_mat.at<double>(2) < 0)
                scale_mat = -scale_mat;
            p_n.push_back(scale_mat);
        }
        
        // Case: d' < 0
        float sqrt_prod_phi = sqrt((d1*d1 - d2*d2) * (d2*d2 - d3*d3)) / ((d1-d3) * d2);
        
        float cos_phi = (d1*d3 - d2*d2) / ((d1-d3) * d2);
        float sin_phi[] = {sqrt_prod_phi, -sqrt_prod_phi, -sqrt_prod_phi, sqrt_prod_phi};
        
        for (int i=0; i<4; i++)
        {
            /*
             R' = |cos(phi), 0,  sin(phi)|
                  |    0,   -1,    0,    |
                  |sin(phi), 0, -cos(phi)|
             */
            
            Mat rotation_prime = Mat::eye(3, 3, CV_64F);
            rotation_prime.at<double>(0, 0) = cos_phi;
            rotation_prime.at<double>(0, 2) = sin_phi[i];
            rotation_prime.at<double>(1, 1) = -1;
            rotation_prime.at<double>(2, 0) = sin_phi[i];
            rotation_prime.at<double>(2, 2) = -cos_phi;
            
            Mat rotation_mat = s * U * rotation_prime * V_tp;
            p_R.push_back(rotation_mat);
            
            /*
             t' =          |  x1 |
                  (d1 + d3)|   0 |
                           |  x3 |
             */
            
            Mat translation_prime = Mat::zeros(3, 1, CV_64F);
            translation_prime.at<double>(0) = x1[i];
            translation_prime.at<double>(2) = x3[i];
            translation_prime *= (d1+d3);
            
            Mat translation_mat = U * translation_prime;
            p_t.push_back(translation_mat / norm(translation_mat));
            
            /*
             n' = | x1 |
                  | 0  |
                  |-x3 |
             */
            
            Mat scale_prime = Mat::zeros(3, 1, CV_64F);
            scale_prime.at<double>(0) = x1[i];
            scale_prime.at<double>(2) = x3[i];
            
            Mat scale_mat = V * scale_prime;
            if (scale_mat.at<double>(2) < 0)
                scale_mat = -scale_mat;
            p_n.push_back(scale_mat);
        }
        
        // Triangulate 3D points for all the 8 possible solutions and find the best R|t:
        int best_trans_idx;
        float max_parallax;
        
        vector<Point3f> best_points;
        vector<bool> best_triangulated_state;
        
        float norm_triangulation_score = ScoreRt(p_R, p_t, ref_keypoints, tar_keypoints, inliers, matches, best_points, max_parallax, best_triangulated_state, best_trans_idx);
                
        if (norm_triangulation_score > TRIANGULATION_NORM_SCORE_H_TH && max_parallax > PARALLAX_MIN_DEGREES)
        {
            p_R.at(best_trans_idx).copyTo(R);
            p_t.at(best_trans_idx).copyTo(t);
            
            points = best_points;
            triangulated_state = best_triangulated_state;
            
            return true;
        }
        
        return false;
    }
    
    // Reference: http://isit.u-clermont1.fr/~ab/Classes/DIKU-3DCV2/Handouts/Lecture16.pdf
    bool Initializer::ReconstructFundamental(PointArray &ref_keypoints, PointArray &tar_keypoints, vector<DMatch> &matches, vector<bool> &inliers, int &num_inliers, Mat &F, Mat &R, Mat &t, vector<Point3f> &points, vector<bool> &triangulated_state)
    {
        // Essential Matrix:
        Mat E = camera_matrix.t() * F * camera_matrix;
        
        // 4 Possible solutions:
        Mat R1, R2, trans;
        
        Mat U, w, V_tp;
        SVD::compute(E, w, U, V_tp);
        
        U.col(2).copyTo(trans);
        trans = trans / norm(trans);
        
        Mat W = Mat(3, 3, CV_64F, Scalar(0));
        W.at<double>(0, 1) = -1;
        W.at<double>(1, 0) = 1;
        W.at<double>(2, 2) = 1;
        
        R1 = U * W * V_tp;
        if (determinant(R1) < 0)
            R1 = -R1;
        
        R2 = U * W.t() * V_tp;
        if (determinant(R2) < 0)
            R2 = -R2;
        
        vector<Mat> p_R, p_t;
        p_R.push_back(R1);
        p_t.push_back(trans);
        
        p_R.push_back(R2);
        p_t.push_back(trans);
        
        p_R.push_back(R1);
        p_t.push_back(-trans);
        
        p_R.push_back(R2);
        p_t.push_back(-trans);
        
        // Triangulate 3D points for all the 8 possible solutions and find the best R|t:
        int best_trans_idx;
        float max_parallax;
        
        vector<Point3f> best_points;
        vector<bool> best_triangulated_state;
        
        float norm_triangulation_score = ScoreRt(p_R, p_t, ref_keypoints, tar_keypoints, inliers, matches, best_points, max_parallax, best_triangulated_state, best_trans_idx);
        
        if (norm_triangulation_score > TRIANGULATION_NORM_SCORE_F_TH && max_parallax > PARALLAX_MIN_DEGREES)
        {
            p_R.at(best_trans_idx).copyTo(R);
            p_t.at(best_trans_idx).copyTo(t);
            
            points = best_points;
            triangulated_state = best_triangulated_state;
            
            return true;
        }
        
        return false;
    }
    
    float Initializer::ScoreRt(vector<Mat> &p_R, vector<Mat> &p_t, const PointArray &ref_keypoints, const PointArray &tar_keypoints, const vector<bool> &inliers, const vector<DMatch> &matches, vector<Point3f> &best_point_cloud, float& best_parallax, vector<bool> &best_triangulated_state, int &best_trans_idx)
    {
        // Assuming p_R elements directly correspond to p_t elemetns
        assert(p_R.size() == p_t.size());
        
        best_trans_idx = -1;
        
        int highest_good_points = 0, sum_good_points = 0;
        best_parallax = -1.0;
        best_point_cloud.clear();
        
        for (int i=0; i<p_R.size(); i++)
        {
            float parallax;
            vector<Point3f> point_cloud;
            vector<bool> triangulated_state;
            
            int num_good_points = CheckRt(p_R[i], p_t[i], ref_keypoints, tar_keypoints, inliers, matches, point_cloud, parallax, triangulated_state);
            sum_good_points += num_good_points;
            
            if (num_good_points > highest_good_points)
            {
                highest_good_points = num_good_points;
                
                best_parallax = parallax;
                best_trans_idx = i;
                best_point_cloud = point_cloud;
                best_triangulated_state = triangulated_state;
            }
        }
        
        if (highest_good_points < TRIANGULATION_MIN_POINTS)
            return 0.0f;
        
        return (1.0f * highest_good_points) / sum_good_points;
    }
    
    int Initializer::CheckRt(Mat &R, Mat &t, const PointArray &ref_keypoints, const PointArray &tar_keypoints, const vector<bool> &inliers, const vector<DMatch> &matches, vector<Point3f> &point_cloud, float& max_parallax, vector<bool> &triangulated_state)
    {
        // Intrinsic parameters (3D->2D) for projection error checking:
        float cam_fx = camera_matrix.at<double>(0, 0);
        float cam_fy = camera_matrix.at<double>(1, 1);
        float cam_cx = camera_matrix.at<double>(0, 2);
        float cam_cy = camera_matrix.at<double>(1, 2);
        
        vector<float> cos_parallaxes;
        triangulated_state = vector<bool>(ref_keypoints.size(), false);
        
        point_cloud.clear();
        cos_parallaxes.reserve(ref_keypoints.size());
        
        // P1 = K[I|0]
        Mat ref_origin = Mat::zeros(3, 1, CV_64F);
        Mat P1 = (Mat_<double>(3, 4) << cam_fx, 0.0,    cam_cx, 0.0,
                                        0.0,    cam_fy, cam_cy, 0.0,
                                        0.0,    0.0,    1.0,    0.0);
        
        // P2 = K[R|t]
        Mat tar_origin = -R.t()*t;
        Mat P2 = (Mat_<double>(3, 4) << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0),
                                        R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1),
                                        R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2));
        P2 = camera_matrix * P2;
        
        int num_good_points = 0;
        for (int i=0; i<ref_keypoints.size(); i++)
        {
            if (!inliers[i])
                continue;
            
            KeyPoint ref_kp, tar_kp;
            ref_kp.pt = ref_keypoints[i];
            tar_kp.pt = tar_keypoints[i];
            
            Mat ref_point_3D = Mat(3, 1, CV_64F, Scalar(0));
            Tracking::Triangulate(ref_kp, tar_kp, P1, P2, ref_point_3D);
            
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
            
            float cos_parallax = ref_normal.dot(tar_normal) / (ref_dist * tar_dist);
            
            // Check that the point is in front of the reference camera:
            if (ref_point_3D.at<double>(2) <= 0.0 && cos_parallax < 0.9998)
            {
                continue;
            }
            
            Mat tar_point_3D = R * ref_point_3D + t;
            
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
            
            // Check reprojection error for target image:b
            float tar_reproj_x, tar_reproj_y;
            tar_reproj_x = cam_fx * (tar_point_3D.at<double>(0) / tar_point_3D.at<double>(2)) + cam_cx;
            tar_reproj_y = cam_fy * (tar_point_3D.at<double>(1) / tar_point_3D.at<double>(2)) + cam_cy;
            
            float tar_square_error = (tar_reproj_x - tar_kp.pt.x) * (tar_reproj_x - tar_kp.pt.x) +
            (tar_reproj_y - tar_kp.pt.y) * (tar_reproj_y - tar_kp.pt.y);
            
            if (tar_square_error > REPROJECTION_ERROR_TH)
            {
                continue;
            }
            
            cos_parallaxes.push_back(cos_parallax);
            point_cloud.push_back(Point3f(ref_point_3D.at<double>(0), ref_point_3D.at<double>(1), ref_point_3D.at<double>(2)));
            
            num_good_points++;
            if (cos_parallax < 0.9998)
                triangulated_state[i] = true;
        }
        
        // Find the max parallax (in degrees) of the first N=TRIANGULATION_MIN_POINTS points
        if (num_good_points > 0)
        {
            sort(cos_parallaxes.begin(), cos_parallaxes.end());
            int nth_max_idx = cos_parallaxes.size()-1 > TRIANGULATION_MIN_POINTS ? TRIANGULATION_MIN_POINTS : (int)cos_parallaxes.size()-1;
            
            max_parallax = acos(cos_parallaxes.at(nth_max_idx)) * 180/CV_PI;
        }
        else
        {
            max_parallax = 0.0;
        }
        
        return num_good_points;
    }
    
    void Initializer::FilterInliers(PointArray &ref_keypoints, PointArray &tar_keypoints, vector<bool> &inliers, PointArray &ref_inliers, PointArray &tar_inliers)
    {
        assert(ref_keypoints != ref_inliers);
        assert(tar_keypoints != tar_inliers);
        
        for (int i=0; i<ref_keypoints.size(); i++)
        {
            if (inliers.at(i))
            {
                ref_inliers.push_back(ref_keypoints.at(i));
                tar_inliers.push_back(tar_keypoints.at(i));
            }
        }
    }
    
    float Initializer::CheckHomography(PointArray& ref_keypoints, PointArray& tar_keypoints, Mat &H_ref2tar, vector<bool> &match_inliers, int &num_inliers)
    {
        float score = 0;
        Mat H_tar2ref = H_ref2tar.inv();
        
        const float inv_sigma_square = 1.0 / (SYMMETRIC_ERROR_SIGMA * SYMMETRIC_ERROR_SIGMA);
        match_inliers.resize(ref_keypoints.size());
        
        num_inliers = 0;
        
        assert(ref_keypoints.size() == tar_keypoints.size());
        for(int i=0; i<ref_keypoints.size(); i++)
        {
            bool is_inlier = true;
            
            const float x1 = ref_keypoints[i].x;
            const float y1 = ref_keypoints[i].y;
            const float x2 = tar_keypoints[i].x;
            const float y2 = tar_keypoints[i].y;
            
            // Reproject tar keypoints to ref keypoints:
            Mat x2_y2 = (Mat_<double>(3, 1) << x2, y2, 1.0f);
            Mat reproj_x1_y1 = H_tar2ref * x2_y2;
            
            const float reproj_w1 = 1.0 / reproj_x1_y1.at<double>(2, 0);
            const float reproj_x1 = reproj_x1_y1.at<double>(0, 0) * reproj_w1;
            const float reproj_y1 = reproj_x1_y1.at<double>(1, 0) * reproj_w1;
            
            // Euclidean distance between 2D points:
            const float ref_square_dist = (x1-reproj_x1)*(x1-reproj_x1) + (y1-reproj_y1)*(y1-reproj_y1);
            const float ref_chi_square = ref_square_dist * inv_sigma_square;
            
            if (ref_chi_square > SYMMETRIC_ERROR_TH)
            {
                is_inlier = false;
            }
            else
            {
                score += SYMMETRIC_ERROR_TH - ref_chi_square;
            }
            
            // Reproject ref keypoints to tar keypoints;
            Mat x1_y1 = (Mat_<double>(3, 1) << x1, y1, 1.0f);
            Mat reproj_x2_y2 = H_ref2tar * x1_y1;
            
            const float reproj_w2 = 1.0 / reproj_x2_y2.at<double>(2, 0);
            const float reproj_x2 = reproj_x2_y2.at<double>(0, 0) * reproj_w2;
            const float reproj_y2 = reproj_x2_y2.at<double>(1, 0) * reproj_w2;
            
            // Euclidean distance between 2D points:
            const float tar_square_dist = (x2-reproj_x2)*(x2-reproj_x2) + (y2-reproj_y2)*(y2-reproj_y2);
            const float tar_chi_square = tar_square_dist * inv_sigma_square;
            
            if (tar_square_dist > SYMMETRIC_ERROR_TH)
            {
                is_inlier = false;
            }
            else
            {
                score += SYMMETRIC_ERROR_TH - tar_chi_square;
            }
            
            // Update inlier status:
            if (is_inlier)
            {
                match_inliers[i] = true;
                num_inliers++;
            }
            else
            {
                match_inliers[i] = false;
            }
        }
        
        return score;
    }
    
    float Initializer::CheckFundamental(PointArray &ref_keypoints, PointArray &tar_keypoints, Mat &F, vector<bool> &match_inliers, int &num_inliers)
    {
        float score = 0;
        
        const float inv_sigma_square = 1.0 / (SYMMETRIC_ERROR_SIGMA * SYMMETRIC_ERROR_SIGMA);
        match_inliers.resize(ref_keypoints.size());
        
        num_inliers = 0;
        
        assert(ref_keypoints.size() == tar_keypoints.size());
        for (int i=0; i<ref_keypoints.size(); i++)
        {
            bool is_inliner = true;
            
            const float x1 = ref_keypoints[i].x;
            const float y1 = ref_keypoints[i].y;
            const float x2 = tar_keypoints[i].x;
            const float y2 = tar_keypoints[i].y;
            
            Mat x1_y1 = (Mat_<double>(3, 1) << x1, y1, 1.0f);
            Mat x2_y2 = (Mat_<double>(3, 1) << x2, y2, 1.0f);
            Mat x1_y1_tp = x1_y1.t();
            Mat x2_y2_tp = x2_y2.t();
            
            // Project ref keypoints to target keypoints (aT * F * a'):
            Mat F_alpha_prime = F * x1_y1;
            Mat alpha_tp_F_alpha_prime = x2_y2_tp * F_alpha_prime;
            
            const float ref_square_dist = (alpha_tp_F_alpha_prime.at<double>(0, 0) * alpha_tp_F_alpha_prime.at<double>(0, 0)) /             ( F_alpha_prime.at<double>(0, 0) * F_alpha_prime.at<double>(0, 0) + F_alpha_prime.at<double>(1, 0) * F_alpha_prime.at<double>(1, 0) );
            
            const float ref_chi_square = ref_square_dist * inv_sigma_square;
            
            if (ref_chi_square > FUNDAMENTAL_ERROR_TH)
            {
                is_inliner = false;
            }
            else
            {
                score += FUNDAMENTAL_ERROR_TH_SCORE - ref_chi_square;
            }
            
            // Project tar keypoints to ref keypoints (a'T * F * a):
            Mat F_alpha = F * x2_y2;
            Mat alpha_prime_tp_F_alpha = x1_y1_tp * F_alpha;
            
            const float tar_square_dist = (alpha_prime_tp_F_alpha.at<double>(0, 0) * alpha_prime_tp_F_alpha.at<double>(0, 0)) /
            (F_alpha.at<double>(0, 0) * F_alpha.at<double>(0, 0) + F_alpha.at<double>(1, 0) * F_alpha.at<double>(1, 0                                                                                                              ));
            
            const float tar_chi_square = tar_square_dist * inv_sigma_square;
            
            if (tar_chi_square > FUNDAMENTAL_ERROR_TH)
            {
                is_inliner = false;
            }
            else
            {
                score += FUNDAMENTAL_ERROR_TH_SCORE - tar_chi_square;
            }
            
            if (is_inliner)
            {
                match_inliers[i] = true;
                num_inliers++;
            }
            else
            {
                match_inliers[i] = false;
            }
            
        }
        
        return score;
    }
    
    void Initializer::Normalize(const PointArray &in_points, PointArray &norm_points, Mat &T)
    {
        float mean_x = 0.0f, mean_y = 0.0f;
        float sum_x = 0.0f, sum_y = 0.0f;
        
        const int num_points = (int)in_points.size();
        
        norm_points.resize(num_points);
        
        for (int i=0; i<num_points; i++)
        {
            sum_x += in_points.at(i).x;
            sum_y += in_points.at(i).y;
        }
        
        mean_x = sum_x / num_points;
        mean_y = sum_y / num_points;
        
        float mean_dev_x = 0.0f, mean_dev_y = 0.0f;
        
        for (int i=0; i<num_points; i++)
        {
            norm_points[i].x += in_points[i].x - mean_x;
            norm_points[i].y += in_points[i].y - mean_y;
            
            mean_dev_x += abs(norm_points[i].x);
            mean_dev_y += abs(norm_points[i].y);
        }
        
        mean_dev_x /= num_points;
        mean_dev_y /= num_points;
        
        float scale_x = 1.0 / mean_dev_x;
        float scale_y = 1.0 / mean_dev_y;
        
        for (int i=0; i<num_points; i++)
        {
            norm_points[i].x *= scale_x;
            norm_points[i].y *= scale_y;
        }
        
        T = Mat::eye(3, 3, CV_64F);
        T.at<double>(0,0) = scale_x;
        T.at<double>(1,1) = scale_y;
        T.at<double>(0,2) = -mean_x * scale_x;
        T.at<double>(1,2) = -mean_y * scale_y;
    }
    
}
