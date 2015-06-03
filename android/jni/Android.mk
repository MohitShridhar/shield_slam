LOCAL_PATH := $(call my-dir)

# Deep Belief
include $(CLEAR_VARS)
LOCAL_MODULE := jpcnn
LOCAL_SRC_FILES := DeepBelief/libjpcnn.so
include $(PREBUILT_SHARED_LIBRARY)

# Tegra optimized OpenCV.mk
include $(CLEAR_VARS)
OPENCV_INSTALL_MODULES:=on
include $(OPENCV_PATH)/sdk/native/jni/OpenCV-tegra3.mk

# Artsy
LOCAL_LDLIBS += -llog
LOCAL_SHARED_LIBRARIES  += jpcnn
LOCAL_MODULE    := Artsy
LOCAL_SRC_FILES := NativeLogging.cpp NativeCore.cpp ss/Augmentor.cpp ss/Common.cpp ss/Initializer.cpp ss/KeyFrame.cpp ss/Optimizer.cpp ss/ORB.cpp ss/Tracking.cpp ss/VSlam.cpp

include $(BUILD_SHARED_LIBRARY)
