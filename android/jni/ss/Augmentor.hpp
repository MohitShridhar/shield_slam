#ifndef __shield_slam__Augmentor__
#define __shield_slam__Augmentor__

#include <opencv2/opencv.hpp>

using namespace cv;

class Augmentor
{
public:
    
    Augmentor();
    
    virtual ~Augmentor() = default;
    
    /**
     * Display translation coordinates in the top left of frame
     */
    void DisplayTranslation(Mat& frame, Mat t);
    
    /**
     * Display rotation coordinates in the bottom left of frame
     */
    void DisplayRotation(Mat& frame, Mat R);
    
    
private:
    double _left_margin;
    double _text_height;
    int _font_face;
    double _font_scale;
    Scalar _font_color_trans;
    Scalar _font_color_rot;
    int _font_thickness;
    int _num_trans_coords;
};


#endif /* defined(__shield_slam__Augmentor__) */
