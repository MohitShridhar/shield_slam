#include <opencv2/core/core.hpp>

#include "../ss/MapPoint.hpp"

using namespace cv;
using namespace std;
using namespace vslam;

class UpdateListener
{
public:
	virtual void update(vector<KeyFrame> keyframes, Mat camera_rot, Mat camera_pos) = 0;
};