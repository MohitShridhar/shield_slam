#include "Augmentor.hpp"
#include <string>

using namespace cv;
using namespace std;

Augmentor::Augmentor() : _left_margin(10.0),
_text_height(20.0),
_font_face(CV_FONT_HERSHEY_PLAIN),
_font_scale(1.0),
_font_color_trans(0, 255 , 0),
_font_color_rot(0, 0, 255),
_font_thickness(2),
_num_trans_coords(3)
{}

void Augmentor::DisplayTranslation(Mat& frame, Mat t)
{
    double t_x = t.at<double>(0);
    double t_y = t.at<double>(1);
    double t_z = t.at<double>(2);
    
    // String streams are necessary because Shield does not have C++11 support
    ostringstream txStringStream;
    ostringstream tyStringStream;
    ostringstream tzStringStream;
    txStringStream << "Translation X: " << t_x;
    tyStringStream << "Translation Y: " << t_y;
    tzStringStream << "Translation Z: " << t_z;
    string txString = txStringStream.str();
    string tyString = tyStringStream.str();
    string tzString = tzStringStream.str();
    
    Point txBottomLeftCoord = Point(_left_margin, 1*_text_height);
    Point tyBottomLeftCoord = Point(_left_margin, 2*_text_height);
    Point tzBottomLeftCoord = Point(_left_margin, 3*_text_height);
    
    // Render XYZ
    putText(frame, txString, txBottomLeftCoord,
            _font_face, _font_scale, _font_color_trans, _font_thickness);
    putText(frame, tyString, tyBottomLeftCoord,
            _font_face, _font_scale, _font_color_trans, _font_thickness);
    putText(frame, tzString, tzBottomLeftCoord,
            _font_face, _font_scale, _font_color_trans, _font_thickness);
}

void Augmentor::DisplayRotation(Mat& frame, Mat R)
{
    // Extract values from R matrix
    double r_11 = R.at<double>(1, 1);
    double r_21 = R.at<double>(2, 1);
    double r_31 = R.at<double>(3, 1);
    double r_32 = R.at<double>(3, 2);
    double r_33 = R.at<double>(3, 3);
    
    // Calculate yaw, pitch and roll
    double yaw = atan(r_21 / r_11);
    double pitch = atan(-r_31 / sqrt(pow(r_32, 2) + pow(r_33, 2)));
    double roll = atan(r_32 / r_33);
    
    ostringstream yawStringStream;
    ostringstream pitchStringStream;
    ostringstream rollStringStream;
    yawStringStream << "Yaw: " << yaw;
    pitchStringStream << "Pitch: " << pitch;
    rollStringStream << "Roll: " << roll;
    string yawString = yawStringStream.str();
    string pitchString = pitchStringStream.str();
    string rollString = rollStringStream.str();
    
    Point yawBottomLeftCoord = Point(_left_margin, (_num_trans_coords + 1)*_text_height);
    Point pitchBottomLeftCoord = Point(_left_margin, (_num_trans_coords + 2)*_text_height);
    Point rollBottomLeftCoord = Point(_left_margin, (_num_trans_coords + 3)*_text_height);
    
    // Render YPR
    putText(frame, yawString, yawBottomLeftCoord,
            _font_face, _font_scale, _font_color_trans, _font_thickness);
    putText(frame, pitchString, pitchBottomLeftCoord,
            _font_face, _font_scale, _font_color_trans, _font_thickness);
    putText(frame, rollString, rollBottomLeftCoord,
            _font_face, _font_scale, _font_color_trans, _font_thickness);
}