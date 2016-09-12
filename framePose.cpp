#include "framePose.h"

FramePose::FramePose(const Eigen::Matrix3f& K, Sim3 otherToThis)
{
    Sim3 thisToOther = otherToThis.inverse();

    //otherToThis = data.worldToCam * other->data.camToWorld;
    K_otherToThis_R = K * otherToThis.rotationMatrix().cast<float>() * otherToThis.scale();
    otherToThis_t = otherToThis.translation().cast<float>();
    K_otherToThis_t = K * otherToThis_t;

    thisToOther_t = thisToOther.translation().cast<float>();
    K_thisToOther_t = K * thisToOther_t;
    thisToOther_R = thisToOther.rotationMatrix().cast<float>() * thisToOther.scale();
    otherToThis_R_row0 = thisToOther_R.col(0);
    otherToThis_R_row1 = thisToOther_R.col(1);
    otherToThis_R_row2 = thisToOther_R.col(2);

    distSquared = otherToThis.translation().dot(otherToThis.translation());
}
