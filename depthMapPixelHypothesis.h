#ifndef DEPTHMAPPIXELHYPOTHESIS
#define DEPTHMAPPIXELHYPOTHESIS

//#pragma once
#include <opencv2/core/core.hpp>
//#include "util/settings.h"
//#include "util/EigenCoreInclude.h"

/** Depth hypothesis used in DepthMap.
 *
 *  Inverse depths need to be scaled with the DepthMap's internalScaleFactor to
 *  get frame scale. (From that, scale with the current keyframe's scale to
 *  get the current best estimate of absolute scale). */
class DepthMapPixelHypothesis
{
public:

    /** Flag telling if there is a valid estimate at this point.
     * All other values are only valid if this is set to true. */
    bool isValid;

    /** Flag that blacklists a point to never be used - set if stereo fails repeatedly on this pixel. */
    int blacklisted;

    /** How many frames to skip ahead in the tracked-frames-queue. */
    float nextStereoFrameMinID;

    /** Counter for validity, basically how many successful observations are incorporated. */
    int validity_counter;

    /** Actual Gaussian Distribution.*/
    float idepth;
    float idepth_var;

    /** Smoothed Gaussian Distribution.*/
    float idepth_smoothed;
    float idepth_var_smoothed;


    inline DepthMapPixelHypothesis() : isValid(false), blacklisted(0) {};

    inline DepthMapPixelHypothesis(
            const float &my_idepth,
            const float &my_idepth_smoothed,
            const float &my_idepth_var,
            const float &my_idepth_var_smoothed,
            const int &my_validity_counter) :
        isValid(true),
        blacklisted(0),
        nextStereoFrameMinID(0),
        validity_counter(my_validity_counter),
        idepth(my_idepth),
        idepth_var(my_idepth_var),
        idepth_smoothed(my_idepth_smoothed),
        idepth_var_smoothed(my_idepth_var_smoothed) {};

    inline DepthMapPixelHypothesis(
            const float &my_idepth,
            const float &my_idepth_var,
            const int &my_validity_counter) :
        isValid(true),
        blacklisted(0),
        nextStereoFrameMinID(0),
        validity_counter(my_validity_counter),
        idepth(my_idepth),
        idepth_var(my_idepth_var),
        idepth_smoothed(-1),
        idepth_var_smoothed(-1) {};

    //cv::Vec3b getVisualizationColor(int lastFrameID) const;
};

#endif // DEPTHMAPPIXELHYPOTHESIS

