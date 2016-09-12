#include "opencv2/core/core.hpp"
#include "slamBase.h"
#include "depthMapPixelHypothesis.h"
#include "globalFunc.h"
#include "framePose.h"

// particularely important for initial pixel.
#define MAX_EPL_LENGTH_CROP 30.0f
#define MIN_EPL_LENGTH_CROP 3.0f

// this is the distance of the sample points used for the stereo descriptor.
#define GRADIENT_SAMPLE_DIST 2.0f //?

// pixel a point needs to be away from border... if too small: segfaults!
#define SAMPLE_POINT_TO_BORDER 7

// pixels with too big an error are definitely thrown out.
#define MAX_ERROR_STEREO (1300.0f) // maximal photometric error for stereo to be successful (sum over 5 squared intensity differences)
#define MIN_DISTANCE_ERROR_STEREO (1.5f) // minimal multiplicative difference to second-best match to not be considered ambiguous.

#define allowNegativeIdepths 1

#define DIVISION_EPS 1e-10f
#define UNZERO(val) (val < 0 ? (val > -1e-10 ? -1e-10 : val) : (val < 1e-10 ? 1e-10 : val))

const float minUseGrad = 5;
const float cameraPixelNoise2 = 4*4;

/**
 * Keeps a detailed depth map (consisting of DepthMapPixelHypothesis) and does
 * stereo comparisons and regularization to update it.
 */

class DepthMap
{
public:

    DepthMap(int w, int h, const Eigen::Matrix3f& K, const cv::Mat _keyFramImage);
    DepthMap(const DepthMap&) = delete;
    DepthMap& operator=(const DepthMap&) = delete;
    ~DepthMap();

    //private:
    // camera matrix etc.
    Eigen::Matrix3f K, KInv;
    float fx,fy,cx,cy;
    float fxi,fyi,cxi,cyi;
    int width, height;

    // ============= internally used buffers for intermediate calculations etc. =============
    // for internal depth tracking, their memory is managed (created & deleted) by this object.
    DepthMapPixelHypothesis* currentDepthMap;

    const float* keyFrameImageData;
    cv::Mat keyFrameImage;

    Eigen::Vector4f* keyFrameGradients;
    bool keyFrameGradientsValid;

    // ============ internal functions ==================================================
    // does the line-stereo seeking.
    // takes a lot of parameters, because they all have been pre-computed before.
    //inline float doLineStereo(
    //        const float u, const float v, const float epxn, const float epyn,
    //        const float min_idepth, const float prior_idepth, float max_idepth,
    //        const Frame* const referenceFrame, const float* referenceFrameImage,
    //        float &result_idepth, float &result_var, float &result_eplLength);
    int doLineStereo(
            const float u, const float v, const float epxn, const float epyn,
            const float min_idepth, const float prior_idepth, float max_idepth,
            FramePose& referenceFrame, const float* referenceFrameImage,
            float &result_idepth, float &result_var, float &result_eplLength);

    void buildGradients();
};
