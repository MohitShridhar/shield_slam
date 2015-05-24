#include <opencv2/core/core.hpp>
#include "MapPoint.hpp"

using namespace cv;
using namespace std;
using namespace vslam;

class UpdateListener
{
public:
	virtual void update(vector<MapPoint> global_map, vector<Mat> camera_rot, vector<Mat> camera_pos) = 0;
};