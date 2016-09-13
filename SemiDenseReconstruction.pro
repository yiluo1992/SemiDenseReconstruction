TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp \
    slamBase.cpp \
    depthEstimation.cpp \
    depthMapPixelHypothesis.cpp \
    depthMap.cpp \
    framePose.cpp \
    sophusUtil.cpp

HEADERS += \
    slamBase.h \
    depthEstimation.h \
    depthMap.h \
    depthMapPixelHypothesis.h \
    framePose.h \
    globalFunc.h \
    sophusUtil.h

INCLUDEPATH += /usr/local/include/opencv2/ /home/yiluo/rosbuild_ws/package_dir/lsd_slam/lsd_slam_core/thirdparty/Sophus/sophus/
INCLUDEPATH += /usr/include/eigen3/ /usr/local/include/pcl-1.8/ /usr/local/include/vtk-6.3/
#For OpenCV
LIBS += /usr/local/lib/libopencv_core.so /usr/local/lib/libopencv_flann.so.2.4
LIBS += /usr/local/lib/libopencv_imgproc.so /usr/local/lib/libopencv_highgui.so /usr/local/lib/libopencv_calib3d.so
LIBS += /usr/local/lib/libopencv_features2d.so  /usr/local/lib/libopencv_nonfree.so
#For PCL
LIBS += /usr/local/lib/libpcl_common.so /usr/local/lib/libpcl_visualization.so /usr/local/lib/libpcl_io.so
LIBS += /usr/local/lib/libpcl_filters.so
#For Boost
LIBS += -lboost_system -lboost_thread
