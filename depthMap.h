#include "opencv2/core/core.hpp"
#include "slamBase.h"
#include "depthMapPixelHypothesis.h"
#include "globalFunc.h"
#include "framePose.h"

/** ============== constants for validity handeling ======================= */
// validity can take values between 0 and X, where X depends on the abs. gradient at that location:
// it is calculated as VALIDITY_COUNTER_MAX + (absGrad/255)*VALIDITY_COUNTER_MAX_VARIABLE
#define VALIDITY_COUNTER_MAX (5.0f)		// validity will never be higher than this
#define VALIDITY_COUNTER_MAX_VARIABLE (250.0f)		// validity will never be higher than this

#define VALIDITY_COUNTER_INC 5		// validity is increased by this on sucessfull stereo
#define VALIDITY_COUNTER_DEC 5		// validity is decreased by this on failed stereo
#define VALIDITY_COUNTER_INITIAL_OBSERVE 5	// initial validity for first observations

/** ============== stereo & gradient calculation ====================== */
// particularely important for initial pixel.
#define MAX_EPL_LENGTH_CROP 50.0f
#define MIN_EPL_LENGTH_CROP 3.0f

// this is the distance of the sample points used for the stereo descriptor.
#define GRADIENT_SAMPLE_DIST 0.5f //?

// pixel a point needs to be away from border... if too small: segfaults!
#define SAMPLE_POINT_TO_BORDER 7

// pixels with too big an error are definitely thrown out.
#define MAX_ERROR_STEREO (30.0f) // maximal photometric error for stereo to be successful (sum over 5 squared intensity differences) 1300.0f
#define MIN_DISTANCE_ERROR_STEREO (2.0f) // minimal multiplicative difference to second-best match to not be considered ambiguous. 1.5

// defines how large the stereo-search region is. it is [mean] +/- [std.dev]*STEREO_EPL_VAR_FAC
#define STEREO_EPL_VAR_FAC 2.0f

// ============== Smoothing and regularization ======================
// distance factor for regularization.
// is used as assumed inverse depth variance between neighbouring pixel.
// basically determines the amount of spacial smoothing (small -> more smoothing).
#define REG_DIST_VAR (0.075f*0.075f*depthSmoothingFactor*depthSmoothingFactor)

// define how strict the merge-processes etc. are.
// are multiplied onto the difference, so the larger, the more restrictive.
#define DIFF_FAC_SMOOTHING (1.0f*1.0f)
#define DIFF_FAC_OBSERVE (1.0f*1.0f)
#define DIFF_FAC_PROP_MERGE (1.0f*1.0f)
#define DIFF_FAC_INCONSISTENT (1.0f * 1.0f)

// ============== initial stereo pixel selection ======================
#define MIN_EPL_GRAD_SQUARED (2.0f*2.0f)
#define MIN_EPL_LENGTH_SQUARED (1.0f*1.0f)
#define MIN_EPL_ANGLE_SQUARED (0.3f*0.3f)

// abs. grad at that location needs to be larger than this.
#define MIN_ABS_GRAD_CREATE (minUseGrad)
#define MIN_ABS_GRAD_DECREASE (minUseGrad)

#define allowNegativeIdepths 1

#define DIVISION_EPS 1e-10f
#define UNZERO(val) (val < 0 ? (val > -1e-10 ? -1e-10 : val) : (val < 1e-10 ? 1e-10 : val))

const float minUseGrad = 5;
const float cameraPixelNoise2 = 4*4;
const float depthSmoothingFactor = 1.0;

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

    float* keyFrameMaxGradients;

    // ============ internal functions ==================================================
    // does the line-stereo seeking.
    // takes a lot of parameters, because they all have been pre-computed before.
    //inline float doLineStereo(
    //        const float u, const float v, const float epxn, const float epyn,
    //        const float min_idepth, const float prior_idepth, float max_idepth,
    //        const Frame* const referenceFrame, const float* referenceFrameImage,
    //        float &result_idepth, float &result_var, float &result_eplLength);
    float doLineStereo(
            const float u, const float v, const float epxn, const float epyn,
            const float min_idepth, const float prior_idepth, float max_idepth,
            FramePose& referenceFrame, const float* referenceFrameImage,
            float &result_idepth, float &result_var, float &result_eplLength);

    void buildGradients();
    void buildMaxGradients();

    void buildDepthMap(const float min_idepth, const float prior_idepth, float max_idepth,
                       FramePose& referenceFrame, const float* referenceFrameImage);
    void updateDepthMap(const float min_idepth, const float prior_idepth, float max_idepth,
                       FramePose& referenceFrame, const float* referenceFrameImage);

    bool observeDepthCreate(const int &x, const int &y, const int &idx,
                                      float min_idepth, float prior_idepth, float max_idepth,
                                      FramePose& referenceFrame, const float* referenceFrameImage);

    bool observeDepthUpdate(const int &x, const int &y, const int &idx,
                                      float min_idepth, float prior_idepth, float max_idepth,
                                      FramePose& referenceFrame, const float* referenceFrameImage);

    bool makeAndCheckEPL(const int x, const int y, const FramePose& ref, float* pepx, float* pepy);

    void regularizeDepthMap();

    void showDepthMap(float min_idepth, float max_idepth);
    void savePointCloud(cv::Mat& referenceFrameImage);
};
