#include "NativeCore.hpp"
#include "NativeLogging.hpp"

#include "ss/Augmentor.hpp"
#include "ss/Common.hpp"
#include "ss/KeyFrame.hpp"
#include "ss/VSlam.hpp"
//#include <string>

//static std::string ConvertString(JNIEnv* env, jstring js)
//{
//    const char* stringChars = env->GetStringUTFChars(js, 0);
//    std::string s = std::string(stringChars);
//    env->ReleaseStringUTFChars(js, stringChars);
//    return s;
//}

JNIEXPORT jlong JNICALL Java_edu_stanford_cvgl_artsy_CameraActivity_CreateNativeController
  (JNIEnv *, jobject)
{
	// Create new VSlam object
	return (jlong)(new vslam::VSlam);
}

JNIEXPORT void JNICALL Java_edu_stanford_cvgl_artsy_CameraActivity_DestroyNativeController
  (JNIEnv *, jobject, jlong addr_native_controller)
{
	delete (vslam::VSlam*)(addr_native_controller);
}

JNIEXPORT void JNICALL Java_edu_stanford_cvgl_artsy_CameraActivity_HandleFrame
  (JNIEnv *, jobject, jlong addr_native_controller, jlong addr_rgba)
{
	// Obtain SLAM object and current camera frame
	vslam::VSlam* slam = (vslam::VSlam*)(addr_native_controller);
	cv::Mat* frame = (cv::Mat*)(addr_rgba);
	cvtColor(*frame, *frame, CV_RGBA2BGR);

	// Update SLAM with the current frame
	slam->ProcessFrame(*frame);

	// Render XYZ and YPR values of the current keyframe
	Augmentor augmentor;
	vslam::KeyFrame currKeyFrame = slam->GetCurrKeyFrame();
  Mat translationMatrix = currKeyFrame.GetTranslation();
  augmentor.DisplayTranslation(*frame, translationMatrix);
  Mat rotationMatrix = currKeyFrame.GetRotation();
  augmentor.DisplayRotation(*frame, rotationMatrix);

	// Render keypoints of the current key frame
	vslam::KeypointArray keypoints = currKeyFrame.GetTrackedKeypoints();
	Scalar kpColor = Scalar(255, 0, 0);
	drawKeypoints(*frame, keypoints, *frame, kpColor);
}

//JNIEXPORT void JNICALL Java_edu_stanford_cvgl_artsy_CameraActivity_SetDataLocation
//  (JNIEnv* env, jobject, jstring path)
//{
//	ar::SetResourceBasePath(ConvertString(env, path));
//}
