#include "depthMap.h"

DepthMap::DepthMap(int w, int h, const Eigen::Matrix3f& K, const cv::Mat _keyFramImage)
{
    width = w;
    height = h;

    this->K = K;
    fx = K(0,0);
    fy = K(1,1);
    cx = K(0,2);
    cy = K(1,2);

    KInv = K.inverse();
    fxi = KInv(0,0);
    fyi = KInv(1,1);
    cxi = KInv(0,2);
    cyi = KInv(1,2);

    currentDepthMap = new DepthMapPixelHypothesis[width*height];
    keyFrameImage = _keyFramImage.clone();
    keyFrameImageData = (float*) keyFrameImage.data;
    keyFrameGradientsValid = false;
    buildGradients();

//    cout << keyFrameImage.row(20);
//    cv::Mat imgShow;
//    keyFrameImage.convertTo(imgShow, CV_8U);
//    cv::imshow("keyframe", imgShow);
//    cv::waitKey();
}

DepthMap::~DepthMap()
{
    delete[] currentDepthMap;

    if(keyFrameGradientsValid)
        delete[] keyFrameGradients;
}

void DepthMap::buildGradients()
{
    keyFrameGradients = new Eigen::Vector4f[width*height];
    const float* img_pt = keyFrameImageData + width;
    const float* img_pt_max = keyFrameImageData + width*(height-1);
    Eigen::Vector4f* gradxyii_pt = keyFrameGradients + width;
    // in each iteration i need -1,0,p1,mw,pw
    float val_m1 = *(img_pt-1);
    float val_00 = *img_pt;
    float val_p1;

    for(; img_pt < img_pt_max; img_pt++, gradxyii_pt++)
    {
        val_p1 = *(img_pt+1);

        *((float*)gradxyii_pt) = 0.5f*(val_p1 - val_m1);
        *(((float*)gradxyii_pt)+1) = 0.5f*(*(img_pt+width) - *(img_pt-width));
        *(((float*)gradxyii_pt)+2) = val_00;

        val_m1 = val_00;
        val_00 = val_p1;
    }
    keyFrameGradientsValid = true;
}

int DepthMap::doLineStereo(
        const float u, const float v, const float epxn, const float epyn,
        const float min_idepth, const float prior_idepth, float max_idepth,
        FramePose& referenceFrame, const float* referenceFrameImage,
        float &result_idepth, float &result_var, float &result_eplLength)
{
    cv::Point2f searchedPoint(-1.0, -1.0);

    // calculate epipolar line start and end point in old image
    Eigen::Vector3f KinvP = Eigen::Vector3f(fxi*u+cxi,fyi*v+cyi,1.0f);
    Eigen::Vector3f pInf = referenceFrame.K_otherToThis_R * KinvP;
    Eigen::Vector3f pReal = pInf / prior_idepth + referenceFrame.K_otherToThis_t;

    float rescaleFactor = pReal[2] * prior_idepth;
    rescaleFactor = 1.0;

    float firstX = u - 2*epxn*rescaleFactor;
    float firstY = v - 2*epyn*rescaleFactor;
    float lastX = u + 2*epxn*rescaleFactor;
    float lastY = v + 2*epyn*rescaleFactor;
    // width - 2 and height - 2 comes from the one-sided gradient calculation at the bottom
    if (firstX <= 0 || firstX >= width - 2
            || firstY <= 0 || firstY >= height - 2
            || lastX <= 0 || lastX >= width - 2
            || lastY <= 0 || lastY >= height - 2) {
        return -1;
    }

    if(!(rescaleFactor > 0.7f && rescaleFactor < 1.4f))
    {
        //if(enablePrintDebugInfo) stats->num_stereo_rescale_oob++;
        return -1;
    }

    // calculate values to search for
    float realVal_p1 = getInterpolatedElement(keyFrameImageData,u + epxn*rescaleFactor, v + epyn*rescaleFactor, width);
    float realVal_m1 = getInterpolatedElement(keyFrameImageData,u - epxn*rescaleFactor, v - epyn*rescaleFactor, width);
    float realVal = getInterpolatedElement(keyFrameImageData,u, v, width);
    float realVal_m2 = getInterpolatedElement(keyFrameImageData,u - 2*epxn*rescaleFactor, v - 2*epyn*rescaleFactor, width);
    float realVal_p2 = getInterpolatedElement(keyFrameImageData,u + 2*epxn*rescaleFactor, v + 2*epyn*rescaleFactor, width);

    Eigen::Vector3f pClose = pInf + referenceFrame.K_otherToThis_t*max_idepth;
    // if the assumed close-point lies behind the
    // image, have to change that.
    if(pClose[2] < 0.001f)
    {
        max_idepth = (0.001f-pInf[2]) / referenceFrame.K_otherToThis_t[2];
        pClose = pInf + referenceFrame.K_otherToThis_t*max_idepth;
    }
    pClose = pClose / pClose[2]; // pos in new image of point (xy), assuming max_idepth

    Eigen::Vector3f pFar = pInf + referenceFrame.K_otherToThis_t*min_idepth;
    // if the assumed far-point lies behind the image or closter than the near-point,
    // we moved past the Point it and should stop.
    if(pFar[2] < 0.001f || max_idepth < min_idepth)
    {
        //if(enablePrintDebugInfo) stats->num_stereo_inf_oob++;
        return -1;
    }
    pFar = pFar / pFar[2]; // pos in new image of point (xy), assuming min_idepth


    // check for nan due to eg division by zero.
    if(isnanf((float)(pFar[0]+pClose[0])))
        return -4;


    // calculate increments in which we will step through the epipolar line.
    // they are sampleDist (or half sample dist) long
    float incx = pClose[0] - pFar[0];
    float incy = pClose[1] - pFar[1];
    float eplLength = sqrt(incx*incx+incy*incy);
    if(!eplLength > 0 || std::isinf(eplLength)) return -4;

    if(eplLength > MAX_EPL_LENGTH_CROP)
    {
        pClose[0] = pFar[0] + incx*MAX_EPL_LENGTH_CROP/eplLength;
        pClose[1] = pFar[1] + incy*MAX_EPL_LENGTH_CROP/eplLength;
    }

    incx *= GRADIENT_SAMPLE_DIST/eplLength;
    incy *= GRADIENT_SAMPLE_DIST/eplLength;

    // extend one sample_dist to left & right.
    pFar[0] -= incx;
    pFar[1] -= incy;
    pClose[0] += incx;
    pClose[1] += incy;


    // make epl long enough (pad a little bit).
    if(eplLength < MIN_EPL_LENGTH_CROP)
    {
        float pad = (MIN_EPL_LENGTH_CROP - (eplLength)) / 2.0f;
        pFar[0] -= incx*pad;
        pFar[1] -= incy*pad;

        pClose[0] += incx*pad;
        pClose[1] += incy*pad;
    }

    // if inf point is outside of image: skip pixel.
    if(
            pFar[0] <= SAMPLE_POINT_TO_BORDER ||
            pFar[0] >= width-SAMPLE_POINT_TO_BORDER ||
            pFar[1] <= SAMPLE_POINT_TO_BORDER ||
            pFar[1] >= height-SAMPLE_POINT_TO_BORDER)
    {
        //if(enablePrintDebugInfo) stats->num_stereo_inf_oob++;
        return -1;
    }



    // if near point is outside: move inside, and test length again.
    if(
            pClose[0] <= SAMPLE_POINT_TO_BORDER ||
            pClose[0] >= width-SAMPLE_POINT_TO_BORDER ||
            pClose[1] <= SAMPLE_POINT_TO_BORDER ||
            pClose[1] >= height-SAMPLE_POINT_TO_BORDER)
    {
        if(pClose[0] <= SAMPLE_POINT_TO_BORDER)
        {
            float toAdd = (SAMPLE_POINT_TO_BORDER - pClose[0]) / incx;
            pClose[0] += toAdd * incx;
            pClose[1] += toAdd * incy;
        }
        else if(pClose[0] >= width-SAMPLE_POINT_TO_BORDER)
        {
            float toAdd = (width-SAMPLE_POINT_TO_BORDER - pClose[0]) / incx;
            pClose[0] += toAdd * incx;
            pClose[1] += toAdd * incy;
        }

        if(pClose[1] <= SAMPLE_POINT_TO_BORDER)
        {
            float toAdd = (SAMPLE_POINT_TO_BORDER - pClose[1]) / incy;
            pClose[0] += toAdd * incx;
            pClose[1] += toAdd * incy;
        }
        else if(pClose[1] >= height-SAMPLE_POINT_TO_BORDER)
        {
            float toAdd = (height-SAMPLE_POINT_TO_BORDER - pClose[1]) / incy;
            pClose[0] += toAdd * incx;
            pClose[1] += toAdd * incy;
        }

        // get new epl length
        float fincx = pClose[0] - pFar[0];
        float fincy = pClose[1] - pFar[1];
        float newEplLength = sqrt(fincx*fincx+fincy*fincy);

        // test again
        if(
                pClose[0] <= SAMPLE_POINT_TO_BORDER ||
                pClose[0] >= width-SAMPLE_POINT_TO_BORDER ||
                pClose[1] <= SAMPLE_POINT_TO_BORDER ||
                pClose[1] >= height-SAMPLE_POINT_TO_BORDER ||
                newEplLength < 8.0f
                )
        {
            //if(enablePrintDebugInfo) stats->num_stereo_near_oob++;
            return -1;
        }
    }

    // from here on:
    // - pInf: search start-point
    // - p0: search end-point
    // - incx, incy: search steps in pixel
    // - eplLength, min_idepth, max_idepth: determines search-resolution, i.e. the result's variance.


    float cpx = pFar[0];
    float cpy =  pFar[1];

    float val_cp_m2 = getInterpolatedElement(referenceFrameImage,cpx-2.0f*incx, cpy-2.0f*incy, width);
    float val_cp_m1 = getInterpolatedElement(referenceFrameImage,cpx-incx, cpy-incy, width);
    float val_cp = getInterpolatedElement(referenceFrameImage,cpx, cpy, width);
    float val_cp_p1 = getInterpolatedElement(referenceFrameImage,cpx+incx, cpy+incy, width);
    float val_cp_p2;



    /*
         * Subsequent exact minimum is found the following way:
         * - assuming lin. interpolation, the gradient of Error at p1 (towards p2) is given by
         *   dE1 = -2sum(e1*e1 - e1*e2)
         *   where e1 and e2 are summed over, and are the residuals (not squared).
         *
         * - the gradient at p2 (coming from p1) is given by
         * 	 dE2 = +2sum(e2*e2 - e1*e2)
         *
         * - linear interpolation => gradient changes linearely; zero-crossing is hence given by
         *   p1 + d*(p2-p1) with d = -dE1 / (-dE1 + dE2).
         *
         *
         *
         * => I for later exact min calculation, I need sum(e_i*e_i),sum(e_{i-1}*e_{i-1}),sum(e_{i+1}*e_{i+1})
         *    and sum(e_i * e_{i-1}) and sum(e_i * e_{i+1}),
         *    where i is the respective winning index.
         */


    // walk in equally sized steps, starting at depth=infinity.
    int loopCounter = 0;
    float best_match_x = -1;
    float best_match_y = -1;
    float best_match_err = 1e50;
    float second_best_match_err = 1e50;

    // best pre and post errors.
    float best_match_errPre=NAN, best_match_errPost=NAN, best_match_DiffErrPre=NAN, best_match_DiffErrPost=NAN;
    bool bestWasLastLoop = false;

    float eeLast = -1; // final error of last comp.

    // alternating intermediate vars
    float e1A=NAN, e1B=NAN, e2A=NAN, e2B=NAN, e3A=NAN, e3B=NAN, e4A=NAN, e4B=NAN, e5A=NAN, e5B=NAN;

    int loopCBest=-1, loopCSecond =-1;
    while(((incx < 0) == (cpx > pClose[0]) && (incy < 0) == (cpy > pClose[1])) || loopCounter == 0)
    {
        // interpolate one new point
        val_cp_p2 = getInterpolatedElement(referenceFrameImage,cpx+2*incx, cpy+2*incy, width);


        // hacky but fast way to get error and differential error: switch buffer variables for last loop.
        float ee = 0;
        if(loopCounter%2==0)
        {
            // calc error and accumulate sums.
            e1A = val_cp_p2 - realVal_p2;ee += e1A*e1A;
            e2A = val_cp_p1 - realVal_p1;ee += e2A*e2A;
            e3A = val_cp - realVal;      ee += e3A*e3A;
            e4A = val_cp_m1 - realVal_m1;ee += e4A*e4A;
            e5A = val_cp_m2 - realVal_m2;ee += e5A*e5A;
        }
        else
        {
            // calc error and accumulate sums.
            e1B = val_cp_p2 - realVal_p2;ee += e1B*e1B;
            e2B = val_cp_p1 - realVal_p1;ee += e2B*e2B;
            e3B = val_cp - realVal;      ee += e3B*e3B;
            e4B = val_cp_m1 - realVal_m1;ee += e4B*e4B;
            e5B = val_cp_m2 - realVal_m2;ee += e5B*e5B;
        }


        // do I have a new winner??
        // if so: set.
        if(ee < best_match_err)
        {
            // put to second-best
            second_best_match_err=best_match_err;
            loopCSecond = loopCBest;

            // set best.
            best_match_err = ee;
            loopCBest = loopCounter;

            best_match_errPre = eeLast;
            best_match_DiffErrPre = e1A*e1B + e2A*e2B + e3A*e3B + e4A*e4B + e5A*e5B;
            best_match_errPost = -1;
            best_match_DiffErrPost = -1;

            best_match_x = cpx;
            best_match_y = cpy;
            bestWasLastLoop = true;
        }
        // otherwise: the last might be the current winner, in which case i have to save these values.
        else
        {
            if(bestWasLastLoop)
            {
                best_match_errPost = ee;
                best_match_DiffErrPost = e1A*e1B + e2A*e2B + e3A*e3B + e4A*e4B + e5A*e5B;
                bestWasLastLoop = false;
            }

            // collect second-best:
            // just take the best of all that are NOT equal to current best.
            if(ee < second_best_match_err)
            {
                second_best_match_err=ee;
                loopCSecond = loopCounter;
            }
        }


        // shift everything one further.
        eeLast = ee;
        val_cp_m2 = val_cp_m1; val_cp_m1 = val_cp; val_cp = val_cp_p1; val_cp_p1 = val_cp_p2;

        //if(enablePrintDebugInfo) stats->num_stereo_comparisons++;

        cpx += incx;
        cpy += incy;

        loopCounter++;
    }

    cout << "Found Point: " << best_match_x << ", " << best_match_y << " With Error: " << best_match_err << endl;

    // if error too big, will return -3, otherwise -2.
//    if(best_match_err > 4.0f*(float)MAX_ERROR_STEREO)
//    {
//        //if(enablePrintDebugInfo) stats->num_stereo_invalid_bigErr++;
//        return -3;
//    }


//    // check if clear enough winner
//    if(abs(loopCBest - loopCSecond) > 1.0f && MIN_DISTANCE_ERROR_STEREO * best_match_err > second_best_match_err)
//    {
//        //if(enablePrintDebugInfo) stats->num_stereo_invalid_unclear_winner++;
//        return -2;
//    }

    bool didSubpixel = false;
    bool useSubpixelStereo = true;
    if(useSubpixelStereo)
    {
        // ================== compute exact match =========================
        // compute gradients (they are actually only half the real gradient)
        float gradPre_pre = -(best_match_errPre - best_match_DiffErrPre);
        float gradPre_this = +(best_match_err - best_match_DiffErrPre);
        float gradPost_this = -(best_match_err - best_match_DiffErrPost);
        float gradPost_post = +(best_match_errPost - best_match_DiffErrPost);

        // final decisions here.
        bool interpPost = false;
        bool interpPre = false;

        // if one is oob: return false.
        //if(enablePrintDebugInfo && (best_match_errPre < 0 || best_match_errPost < 0))
        if((best_match_errPre < 0 || best_match_errPost < 0))
        //{
        //    stats->num_stereo_invalid_atEnd++;
        //}
        {

        }
        // - if zero-crossing occurs exactly in between (gradient Inconsistent),
        else if((gradPost_this < 0) ^ (gradPre_this < 0))
        {
            // return exact pos, if both central gradients are small compared to their counterpart.
            //if(enablePrintDebugInfo && (gradPost_this*gradPost_this > 0.1f*0.1f*gradPost_post*gradPost_post ||
            //                            gradPre_this*gradPre_this > 0.1f*0.1f*gradPre_pre*gradPre_pre))
            //    stats->num_stereo_invalid_inexistantCrossing++;
        }

        // if pre has zero-crossing
        else if((gradPre_pre < 0) ^ (gradPre_this < 0))
        {
            // if post has zero-crossing
            if((gradPost_post < 0) ^ (gradPost_this < 0))
            {
                //if(enablePrintDebugInfo) stats->num_stereo_invalid_twoCrossing++;
            }
            else
                interpPre = true;
        }

        // if post has zero-crossing
        else if((gradPost_post < 0) ^ (gradPost_this < 0))
        {
            interpPost = true;
        }

        // if none has zero-crossing
        else
        {
            //if(enablePrintDebugInfo) stats->num_stereo_invalid_noCrossing++;
        }


        // DO interpolation!
        // minimum occurs at zero-crossing of gradient, which is a straight line => easy to compute.
        // the error at that point is also computed by just integrating.
        if(interpPre)
        {
            float d = gradPre_this / (gradPre_this - gradPre_pre);
            best_match_x -= d*incx;
            best_match_y -= d*incy;
            best_match_err = best_match_err - 2*d*gradPre_this - (gradPre_pre - gradPre_this)*d*d;
            //if(enablePrintDebugInfo) stats->num_stereo_interpPre++;
            didSubpixel = true;

        }
        else if(interpPost)
        {
            float d = gradPost_this / (gradPost_this - gradPost_post);
            best_match_x += d*incx;
            best_match_y += d*incy;
            best_match_err = best_match_err + 2*d*gradPost_this + (gradPost_post - gradPost_this)*d*d;
            //if(enablePrintDebugInfo) stats->num_stereo_interpPost++;
            didSubpixel = true;
        }
        else
        {
            //if(enablePrintDebugInfo) stats->num_stereo_interpNone++;
        }
    }

    // sampleDist is the distance in pixel at which the realVal's were sampled
    float sampleDist = GRADIENT_SAMPLE_DIST*rescaleFactor;

    float gradAlongLine = 0;
    float tmp = realVal_p2 - realVal_p1;  gradAlongLine+=tmp*tmp;
    tmp = realVal_p1 - realVal;  gradAlongLine+=tmp*tmp;
    tmp = realVal - realVal_m1;  gradAlongLine+=tmp*tmp;
    tmp = realVal_m1 - realVal_m2;  gradAlongLine+=tmp*tmp;

    gradAlongLine /= sampleDist*sampleDist;

    // check if interpolated error is OK. use evil hack to allow more error if there is a lot of gradient.
    //if(best_match_err > (float)MAX_ERROR_STEREO + sqrtf( gradAlongLine)*20)
    //{
        //if(enablePrintDebugInfo) stats->num_stereo_invalid_bigErr++;
    //    return -3;
    //}

    // ================= calc depth (in KF) ====================
    // * KinvP = Kinv * (x,y,1); where x,y are pixel coordinates of point we search for, in the KF.
    // * best_match_x = x-coordinate of found correspondence in the reference frame.

    float idnew_best_match;	// depth in the new image
    float alpha; // d(idnew_best_match) / d(disparity in pixel) == conputed inverse depth derived by the pixel-disparity.
    if(incx*incx>incy*incy)
    {
        float oldX = fxi*best_match_x+cxi;
        float nominator = (oldX*referenceFrame.otherToThis_t[2] - referenceFrame.otherToThis_t[0]);
        float dot0 = KinvP.dot(referenceFrame.otherToThis_R_row0);
        float dot2 = KinvP.dot(referenceFrame.otherToThis_R_row2);

        idnew_best_match = (dot0 - oldX*dot2) / nominator;
        alpha = incx*fxi*(dot0*referenceFrame.otherToThis_t[2] - dot2*referenceFrame.otherToThis_t[0]) / (nominator*nominator);

    }
    else
    {
        float oldY = fyi*best_match_y+cyi;

        float nominator = (oldY*referenceFrame.otherToThis_t[2] - referenceFrame.otherToThis_t[1]);
        float dot1 = KinvP.dot(referenceFrame.otherToThis_R_row1);
        float dot2 = KinvP.dot(referenceFrame.otherToThis_R_row2);

        idnew_best_match = (dot1 - oldY*dot2) / nominator;
        alpha = incy*fyi*(dot1*referenceFrame.otherToThis_t[2] - dot2*referenceFrame.otherToThis_t[1]) / (nominator*nominator);

    }

    if(idnew_best_match < 0)
    {
        //if(enablePrintDebugInfo) stats->num_stereo_negative++;
        if(!allowNegativeIdepths)
            return -2;
    }

    result_idepth = idnew_best_match;
    result_eplLength = eplLength;

    //if(enablePrintDebugInfo) stats->num_stereo_successfull++;

    // ================= calc var (in NEW image) ====================

    // calculate error from photometric noise
    float photoDispError = 4.0f * cameraPixelNoise2 / (gradAlongLine + DIVISION_EPS);
    float initialTrackedResidual = 1.0f;
    float trackingErrorFac = 0.25f*(1.0f + initialTrackedResidual);

    // calculate error from geometric noise (wrong camera pose / calibration)
    Eigen::Vector2f gradsInterp = getInterpolatedElement42(keyFrameGradients, u, v, width);
    float geoDispError = (gradsInterp[0]*epxn + gradsInterp[1]*epyn) + DIVISION_EPS;
    geoDispError = trackingErrorFac*trackingErrorFac*(gradsInterp[0]*gradsInterp[0] + gradsInterp[1]*gradsInterp[1]) / (geoDispError*geoDispError);

    //geoDispError *= (0.5 + 0.5 *result_idepth) * (0.5 + 0.5 *result_idepth);

    // final error consists of a small constant part (discretization error),
    // geometric and photometric error.
    result_var = alpha*alpha*((didSubpixel ? 0.05f : 0.5f)*sampleDist*sampleDist +  geoDispError + photoDispError);	// square to make variance

    return -1;
}
