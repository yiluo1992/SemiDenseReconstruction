#ifndef GLOBALFUNC
#define GLOBALFUNC
#include <opencv2/core/core.hpp>
#include "sophusUtil.h"

// reads interpolated element from a uchar* array
// SSE2 optimization possible
inline float getInterpolatedElement(const float* const mat, const float x, const float y, const int width)
{
    //stats.num_pixelInterpolations++;

    int ix = (int)x;
    int iy = (int)y;
    float dx = x - ix;
    float dy = y - iy;
    float dxdy = dx*dy;
    const float* bp = mat +ix+iy*width;


    float res =   dxdy * bp[1+width]
                + (dy-dxdy) * bp[width]
                + (dx-dxdy) * bp[1]
                + (1-dx-dy+dxdy) * bp[0];

    return res;
}

inline Eigen::Vector2f getInterpolatedElement42(const Eigen::Vector4f* const mat, const float x, const float y, const int width)
{
    int ix = (int)x;
    int iy = (int)y;
    float dx = x - ix;
    float dy = y - iy;
    float dxdy = dx*dy;
    const Eigen::Vector4f* bp = mat +ix+iy*width;

    return dxdy * *(const Eigen::Vector2f*)(bp+1+width)
            + (dy-dxdy) * *(const Eigen::Vector2f*)(bp+width)
            + (dx-dxdy) * *(const Eigen::Vector2f*)(bp+1)
            + (1-dx-dy+dxdy) * *(const Eigen::Vector2f*)(bp);
}

#endif // GLOBALFUNC

