#ifndef DEPTHESTIMATION
#define DEPTHESTIMATION

#include <iostream>
#include "slamBase.h"
using namespace std;

// OpenCV
#include <features2d/features2d.hpp>
#include <nonfree/nonfree.hpp>
#include <calib3d/calib3d.hpp>
#include <imgproc/imgproc.hpp>

// PCL
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

// Esimate Depth From Epipolar Constraint
void depthFromSparseMatch();
void searchEpipolarLine();
void buildDepthMapandDispaly();
void fuseMultipleFrames();

#endif // DEPTHESTIMATION

