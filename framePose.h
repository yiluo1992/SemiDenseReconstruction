#ifndef FRAMEPOSE
#define FRAMEPOSE

#include "slamBase.h"
#include "so3.hpp"
#include "se3.hpp"
#include "sim3.hpp"
#include "sophusUtil.h"

class FramePose
{
public:
    FramePose(const Eigen::Matrix3f& K, Sim3 thisToOther);
    //~Frame();

    // Temporary values
    float distSquared;
    Eigen::Matrix3f K_otherToThis_R;
    Eigen::Vector3f K_otherToThis_t;
    Eigen::Vector3f otherToThis_t;
    Eigen::Vector3f K_thisToOther_t;
    Eigen::Matrix3f thisToOther_R;
    Eigen::Vector3f otherToThis_R_row0;
    Eigen::Vector3f otherToThis_R_row1;
    Eigen::Vector3f otherToThis_R_row2;
    Eigen::Vector3f thisToOther_t;
};

#endif // FRAMEPOSE

