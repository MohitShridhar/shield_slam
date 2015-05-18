#include <opencv2/core/core.hpp>

class UpdateListener
{
public:
	virtual void update(std::vector<cv::Point3d> pcld,
						std::vector<cv::Matx34d> cameras) = 0;
};