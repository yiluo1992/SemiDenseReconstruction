#include "depthEstimation.h"
#include "framePose.h"
#include "depthMap.h"

void depthFromSparseMatch()
{
    ParameterReader pd;

    // 声明并从data文件夹里读取两个rgb与深度图
    cv::Mat rgb1 = cv::imread( "./rgb1.png");
    cv::Mat rgb2 = cv::imread( "./rgb3.png");
    cv::Mat depth1 = cv::imread( "./depth1.png", -1);
    cv::Mat depth2 = cv::imread( "./depth2.png", -1);

    // 声明特征提取器与描述子提取器
    cv::Ptr<cv::FeatureDetector> _detector;
    cv::Ptr<cv::DescriptorExtractor> _descriptor;

    // 构建提取器，默认两者都为sift
    // 构建sift, surf之前要初始化nonfree模块
    cv::initModule_nonfree();
    _detector = cv::FeatureDetector::create( "SIFT" );
    _descriptor = cv::DescriptorExtractor::create( "SIFT" );

    vector< cv::KeyPoint > kp1, kp2; //关键点
    _detector->detect( rgb1, kp1 );  //提取关键点
    _detector->detect( rgb2, kp2 );
    // 计算描述子
    cv::Mat desp1, desp2;
    _descriptor->compute( rgb1, kp1, desp1 );
    _descriptor->compute( rgb2, kp2, desp2 );

    // 匹配描述子
    vector< cv::DMatch > matches;
    cv::FlannBasedMatcher matcher;
    matcher.match( desp1, desp2, matches );
    cout<<"Find total "<<matches.size()<<" matches."<<endl;

    // 筛选匹配，把距离太大的去掉
    // 这里使用的准则是去掉大于四倍最小距离的匹配
    vector< cv::DMatch > goodMatches;
    double minDis = 9999;
    for ( size_t i=0; i<matches.size(); i++ )
    {
        if ( matches[i].distance < minDis )
            minDis = matches[i].distance;
    }

    for ( size_t i=0; i<matches.size(); i++ )
    {
        if (matches[i].distance < 4*minDis)
            goodMatches.push_back( matches[i] );
    }

    cv::Mat imgMatches;
//    // 显示 good matches

//    cout<<"good matches="<<goodMatches.size()<<endl;
//    cv::drawMatches( rgb1, kp1, rgb2, kp2, goodMatches, imgMatches );
//    cv::imshow( "good matches", imgMatches );
//    cv::imwrite( "./data/good_matches.png", imgMatches );
//    cv::waitKey(0);

    // 计算图像间的运动关系
    // 关键函数：cv::solvePnPRansac()
    // 为调用此函数准备必要的参数

    // 第一个帧的三维点
    vector<cv::Point3f> pts_obj;
    // 第二个帧的图像点
    vector< cv::Point2f > pts_img;

    // 相机内参
    CAMERA_INTRINSIC_PARAMETERS C;
    C.fx = atof( pd.getData( "camera.fx" ).c_str());
    C.fy = atof( pd.getData( "camera.fy" ).c_str());
    C.cx = atof( pd.getData( "camera.cx" ).c_str());
    C.cy = atof( pd.getData( "camera.cy" ).c_str());
    C.scale = atof( pd.getData( "camera.scale" ).c_str() );

    vector< cv::DMatch > filteredGoodMatches;

    for (size_t i=0; i<goodMatches.size(); i++)
    {
        // query 是第一个, train 是第二个
        cv::Point2f p = kp1[goodMatches[i].queryIdx].pt;
        // 获取d是要小心！x是向右的，y是向下的，所以y才是行，x是列！
        ushort d = depth1.ptr<ushort>( int(p.y) )[ int(p.x) ];
        if (d == 0)
            continue;
        //cout << d << endl;
        pts_img.push_back( cv::Point2f( kp2[goodMatches[i].trainIdx].pt ) );

        // 将(u,v,d)转成(x,y,z)
        cv::Point3f pt ( p.x, p.y, d );
        cv::Point3f pd = point2dTo3d( pt, C );
        pts_obj.push_back( pd );
        filteredGoodMatches.push_back(goodMatches[i]);
    }

    double camera_matrix_data[3][3] = {
        {C.fx, 0, C.cx},
        {0, C.fy, C.cy},
        {0, 0, 1}
    };

    // 构建相机矩阵
    cv::Mat cameraMatrix( 3, 3, CV_64F, camera_matrix_data );
    cv::Mat rvec, tvec, inliers;
    // 求解pnp
    cv::solvePnPRansac( pts_obj, pts_img, cameraMatrix, cv::Mat(), rvec, tvec, false, 100, 0.5, 100, inliers );

    cout<<"inliers: "<<inliers.rows<<endl;
    cout<<"R="<<rvec<<endl;
    cout<<"t="<<tvec<<endl;

    // 画出inliers匹配
    vector< cv::DMatch > matchesShow;
    for (size_t i=0; i<inliers.rows; i++)
    //for (size_t i=0; i<3; i++)
    {
        matchesShow.push_back( filteredGoodMatches[inliers.ptr<int>(i)[0]] );
    }
    cv::drawMatches( rgb1, kp1, rgb2, kp2, matchesShow, imgMatches );
    cv::imshow( "inlier matches", imgMatches );
    cv::imwrite( "./data/inliers.png", imgMatches );
    cv::waitKey( 0 );

    cv::Mat R;
    cv::Rodrigues(rvec, R );
    cout << "Rotation Matrix: " << R << endl;
    cout << "Translation Vector: " << tvec << endl;
    cout << "Inverse K: " << cameraMatrix.inv() << endl;
    cv::Mat inverseCameraMatrix = cameraMatrix.inv();

//    // Get relative pose from ground true
//    // CamToWorld
//    // 1341847980.8201 -0.6821 2.6914 1.7371 0.0003 0.8609 -0.5085 -0.0151
//    // 1341847980.9200 -0.6785 2.6925 1.7375 -0.0003 0.8607 -0.5088 -0.0160
//    Eigen::Quaterniond q1(0.0003, 0.8609, -0.5085, -0.0151);
//    Eigen::Matrix3d R1 = q1.toRotationMatrix().transpose(); // convert a quaternion to a 3x3 rotation matrix
//    cout << "R1:" << R1 << endl;
//    Eigen::Quaterniond q2(-0.0003, 0.8607, -0.5088, -0.0160);
//    Eigen::Matrix3d R2 = q2.toRotationMatrix().transpose(); // convert a quaternion to a 3x3 rotation matrix
//    cout << "R2:" << R2 << endl;
//    cv::Mat R1_OpenCV(R1.rows(), R1.cols(), CV_64F, R1.data());
//    cv::Mat R2_OpenCV(R2.rows(), R2.cols(), CV_64F, R2.data());
//    cv::Mat M1(4, 4, CV_64F, cv::Scalar(0.0));
//    cv::Mat M2(4, 4, CV_64F, cv::Scalar(0.0));
//    // Assign M1
//    M1.at<double>(0,0) = R1_OpenCV.at<double>(0,0);
//    M1.at<double>(0,1) = R1_OpenCV.at<double>(0,1);
//    M1.at<double>(0,2) = R1_OpenCV.at<double>(0,2);
//    M1.at<double>(1,0) = R1_OpenCV.at<double>(1,0);
//    M1.at<double>(1,1) = R1_OpenCV.at<double>(1,1);
//    M1.at<double>(1,2) = R1_OpenCV.at<double>(1,2);
//    M1.at<double>(2,0) = R1_OpenCV.at<double>(2,0);
//    M1.at<double>(2,1) = R1_OpenCV.at<double>(2,1);
//    M1.at<double>(2,2) = R1_OpenCV.at<double>(2,2);
//    M1.at<double>(0,3) = -0.6821;
//    M1.at<double>(1,3) = 2.6914;
//    M1.at<double>(2,3) = 1.7371;
//    M1.at<double>(3,3) = 1.0;
//    cout << "M1" << M1 << endl;
//    // Assign M2
//    M2.at<double>(0,0) = R2_OpenCV.at<double>(0,0);
//    M2.at<double>(0,1) = R2_OpenCV.at<double>(0,1);
//    M2.at<double>(0,2) = R2_OpenCV.at<double>(0,2);
//    M2.at<double>(1,0) = R2_OpenCV.at<double>(1,0);
//    M2.at<double>(1,1) = R2_OpenCV.at<double>(1,1);
//    M2.at<double>(1,2) = R2_OpenCV.at<double>(1,2);
//    M2.at<double>(2,0) = R2_OpenCV.at<double>(2,0);
//    M2.at<double>(2,1) = R2_OpenCV.at<double>(2,1);
//    M2.at<double>(2,2) = R2_OpenCV.at<double>(2,2);
//    M2.at<double>(0,3) = -0.6785;
//    M2.at<double>(1,3) = 2.6925;
//    M2.at<double>(2,3) = 1.7375;
//    M2.at<double>(3,3) = 1.0;
//    cout << "M2" << M2 << endl;

//    cv::Mat M21 = M1*M2.inv();
//    cout << M21 << endl;
//    // Assign R and t
//    R = M21.rowRange(0,3).colRange(0,3);
//    tvec = M21.col(3).rowRange(0,3);
//    cout << R << endl << tvec << endl;

    cv::Mat r_x = R.row(0);
    cv::Mat r_z = R.row(2);
    double fx = C.fx;
    double cx = C.cx;
    double tx = tvec.at<double>(0);
    double tz = tvec.at<double>(2);
    double fxi = inverseCameraMatrix.at<double>(0,0);
    double cxi = inverseCameraMatrix.at<double>(0,2);

    // Probabilistic Semi-Dense
    //for (size_t i=0; i<10; i++)
    for (size_t i=0; i<inliers.rows; i++)
    {
        cv::Point2f pKeyframe = kp1[matchesShow[i].queryIdx].pt;
        cout << "Keyframe point: " << pKeyframe.x << ", " << pKeyframe.y << endl;
        cv::Point2f pSearched = kp2[matchesShow[i].trainIdx].pt;
        cout << "Searched point: " << pSearched.x << ", " << pSearched.y << endl;
        cv::Mat x_p(3, 1, CV_64F);
        x_p.at<double>(0) = pKeyframe.x;
        x_p.at<double>(1) = pKeyframe.y;
        x_p.at<double>(2) = 1.0;
        cv::Mat X_hat_p = inverseCameraMatrix*x_p;
        //cout << X_hat_p << endl;
        double u_j = pSearched.x;
        cv::Mat denominatorVec = r_z*X_hat_p*(u_j - cx) - fx*r_x*X_hat_p;
        double denominator = denominatorVec.at<double>(0);
        double nominator = -tz*(u_j - cx) + fx*tx;
        double estimatedDepth = denominator/nominator;
        //cout << nominator << " " << denominator << endl;
        cout << "estimatedDepth: " << 1.0/estimatedDepth << " trueDepth: " << depth1.ptr<ushort>( int(pKeyframe.y) )[ int(pKeyframe.x) ]/(double)C.scale << endl << endl;
    }

    // LSD SLAM
//    float oldX = fxi*best_match_x+cxi;
//    float nominator = (oldX*referenceFrame->otherToThis_t[2] - referenceFrame->otherToThis_t[0]);
//    float dot0 = KinvP.dot(referenceFrame->otherToThis_R_row0);
//    float dot2 = KinvP.dot(referenceFrame->otherToThis_R_row2);
//    idnew_best_match = (dot0 - oldX*dot2) / nominator;
//    for (size_t i=0; i<3; i++)
//    {
//        cv::Point2f pKeyframe = kp1[matchesShow[i].queryIdx].pt;
//        cout << "Keyframe point: " << pKeyframe.x << ", " << pKeyframe.y << endl;
//        cv::Point2f pSearched = kp2[matchesShow[i].trainIdx].pt;
//        cout << "Searched point: " << pSearched.x << ", " << pSearched.y << endl;
//        cv::Mat x_p(3, 1, CV_64F);
//        x_p.at<double>(0) = pKeyframe.x;
//        x_p.at<double>(1) = pKeyframe.y;
//        x_p.at<double>(2) = 1.0;
//        cv::Mat X_hat_p = inverseCameraMatrix*x_p;
//        //cout << X_hat_p << endl;
//        double u_j = pSearched.x;
//        double X_hat_j_x = fxi*u_j + cxi;
//        cv::Mat denominatorVec = r_x*X_hat_p - X_hat_j_x*r_z*X_hat_p;
//        double denominator = denominatorVec.at<double>(0);
//        double nominator = tz*X_hat_j_x - tx;
//        double estimatedDepth = denominator/nominator;
//        cout << denominator << " " << nominator << endl;
//        cout << "estimatedDepth: " << 1.0/estimatedDepth  << " trueDepth: " << depth1.ptr<ushort>( int(pKeyframe.y) )[ int(pKeyframe.x) ] << endl;
//    }

    /**
    // Get relative pose from ground true
    // 1341847980.8201 -0.6821 2.6914 1.7371 0.0003 0.8609 -0.5085 -0.0151
    // 1341847980.9200 -0.6785 2.6925 1.7375 -0.0003 0.8607 -0.5088 -0.0160
    Eigen::Quaterniond q1(0.0003, 0.8609, -0.5085, -0.0151);
    Eigen::Matrix3d R1 = q1.toRotationMatrix().transpose(); // convert a quaternion to a 3x3 rotation matrix
    cout << "R1:" << R1 << endl;
    Eigen::Quaterniond q2(-0.0003, 0.8607, -0.5088, -0.0160);
    Eigen::Matrix3d R2 = q2.toRotationMatrix().transpose(); // convert a quaternion to a 3x3 rotation matrix
    cout << "R2:" << R2 << endl;
    cv::Mat R1_OpenCV(R1.rows(), R1.cols(), CV_64F, R1.data());
    cv::Mat R2_OpenCV(R2.rows(), R2.cols(), CV_64F, R2.data());
    cv::Mat M1(4, 4, CV_64F, cv::Scalar(0.0));
    cv::Mat M2(4, 4, CV_64F, cv::Scalar(0.0));
    // Assign M1

    cv::Mat Ow1(3, 1, CV_64F);
    Ow1.at<double>(0) = -0.6821;
    Ow1.at<double>(1) = 2.6914;
    Ow1.at<double>(2) = 1.7371;
    cv::Mat t1 = -1.0*R1_OpenCV*Ow1;
    cout << "R1:" << R1_OpenCV << endl;
    cout << "t1: " << t1 << endl;
    M1.at<double>(0,0) = R1_OpenCV.at<double>(0,0);
    M1.at<double>(0,1) = R1_OpenCV.at<double>(0,1);
    M1.at<double>(0,2) = R1_OpenCV.at<double>(0,2);
    M1.at<double>(1,0) = R1_OpenCV.at<double>(1,0);
    M1.at<double>(1,1) = R1_OpenCV.at<double>(1,1);
    M1.at<double>(1,2) = R1_OpenCV.at<double>(1,2);
    M1.at<double>(2,0) = R1_OpenCV.at<double>(2,0);
    M1.at<double>(2,1) = R1_OpenCV.at<double>(2,1);
    M1.at<double>(2,2) = R1_OpenCV.at<double>(2,2);
    M1.at<double>(0,3) = t1.at<double>(0);
    M1.at<double>(1,3) = t1.at<double>(1);
    M1.at<double>(2,3) = t1.at<double>(2);
    M1.at<double>(3,3) = 1.0;
    cout << "M1" << M1 << endl;
    // Assign M2
    cv::Mat Ow2(3, 1, CV_64F);
    Ow2.at<double>(0) = -0.6785;
    Ow2.at<double>(1) = 2.6925;
    Ow2.at<double>(2) = 1.7375;
    cv::Mat t2 = -1.0*R2_OpenCV*Ow2;
    cout << "R2:" << R2_OpenCV << endl;
    cout << "t2: " << t2 << endl;
    M2.at<double>(0,0) = R2_OpenCV.at<double>(0,0);
    M2.at<double>(0,1) = R2_OpenCV.at<double>(0,1);
    M2.at<double>(0,2) = R2_OpenCV.at<double>(0,2);
    M2.at<double>(1,0) = R2_OpenCV.at<double>(1,0);
    M2.at<double>(1,1) = R2_OpenCV.at<double>(1,1);
    M2.at<double>(1,2) = R2_OpenCV.at<double>(1,2);
    M2.at<double>(2,0) = R2_OpenCV.at<double>(2,0);
    M2.at<double>(2,1) = R2_OpenCV.at<double>(2,1);
    M2.at<double>(2,2) = R2_OpenCV.at<double>(2,2);
    M2.at<double>(0,3) = t2.at<double>(0);
    M2.at<double>(1,3) = t2.at<double>(1);
    M2.at<double>(2,3) = t2.at<double>(2);
    M2.at<double>(3,3) = 1.0;
    cout << "M2" << M2 << endl;
      */


}

void searchEpipolarLine()
{
    ParameterReader pd;

    // 声明并从data文件夹里读取两个rgb与深度图
    cv::Mat rgb1 = cv::imread( "./rgb1.png");
    cv::Mat rgb2 = cv::imread( "./rgb3.png");
    cv::Mat depth1 = cv::imread( "./depth1.png", -1);
    cv::Mat depth2 = cv::imread( "./depth2.png", -1);

    // 声明特征提取器与描述子提取器
    cv::Ptr<cv::FeatureDetector> _detector;
    cv::Ptr<cv::DescriptorExtractor> _descriptor;

    // 构建提取器，默认两者都为sift
    // 构建sift, surf之前要初始化nonfree模块
    cv::initModule_nonfree();
    _detector = cv::FeatureDetector::create( "SIFT" );
    _descriptor = cv::DescriptorExtractor::create( "SIFT" );

    vector< cv::KeyPoint > kp1, kp2; //关键点
    _detector->detect( rgb1, kp1 );  //提取关键点
    _detector->detect( rgb2, kp2 );
    // 计算描述子
    cv::Mat desp1, desp2;
    _descriptor->compute( rgb1, kp1, desp1 );
    _descriptor->compute( rgb2, kp2, desp2 );

    // 匹配描述子
    vector< cv::DMatch > matches;
    cv::FlannBasedMatcher matcher;
    matcher.match( desp1, desp2, matches );
    cout<<"Find total "<<matches.size()<<" matches."<<endl;

    // 筛选匹配，把距离太大的去掉
    // 这里使用的准则是去掉大于四倍最小距离的匹配
    vector< cv::DMatch > goodMatches;
    double minDis = 9999;
    for ( size_t i=0; i<matches.size(); i++ )
    {
        if ( matches[i].distance < minDis )
            minDis = matches[i].distance;
    }

    for ( size_t i=0; i<matches.size(); i++ )
    {
        if (matches[i].distance < 4*minDis)
            goodMatches.push_back( matches[i] );
    }

    cv::Mat imgMatches;

    // 显示 good matches
    // 计算图像间的运动关系
    // 关键函数：cv::solvePnPRansac()
    // 为调用此函数准备必要的参数

    // 第一个帧的三维点
    vector<cv::Point3f> pts_obj;
    // 第二个帧的图像点
    vector< cv::Point2f > pts_img;

    // 相机内参
    CAMERA_INTRINSIC_PARAMETERS C;
    C.fx = atof( pd.getData( "camera.fx" ).c_str());
    C.fy = atof( pd.getData( "camera.fy" ).c_str());
    C.cx = atof( pd.getData( "camera.cx" ).c_str());
    C.cy = atof( pd.getData( "camera.cy" ).c_str());
    C.scale = atof( pd.getData( "camera.scale" ).c_str() );

    // filter out those invalid point, depth is 0
    vector< cv::DMatch > filteredGoodMatches;
    // keep track of the mindepth and maxdepth
    float minDepth = FLT_MAX;
    float maxDepth = FLT_MIN;
    for (size_t i=0; i<goodMatches.size(); i++)
    {
        // query 是第一个, train 是第二个
        cv::Point2f p = kp1[goodMatches[i].queryIdx].pt;
        // 获取d是要小心！x是向右的，y是向下的，所以y才是行，x是列！
        ushort d = depth1.ptr<ushort>( int(p.y) )[ int(p.x) ];
        if (d == 0)
            continue;

        float curDepth = (float)d/C.scale;
        if(minDepth > curDepth) minDepth = curDepth;
        if(maxDepth < curDepth) maxDepth = curDepth;

        pts_img.push_back( cv::Point2f( kp2[goodMatches[i].trainIdx].pt ) );

        // 将(u,v,d)转成(x,y,z)
        cv::Point3f pt ( p.x, p.y, d );
        cv::Point3f pd = point2dTo3d( pt, C );
        pts_obj.push_back( pd );
        filteredGoodMatches.push_back(goodMatches[i]);
    }

    double camera_matrix_data[3][3] = {
        {C.fx, 0, C.cx},
        {0, C.fy, C.cy},
        {0, 0, 1}
    };

    // 构建相机矩阵
    cv::Mat cameraMatrix( 3, 3, CV_64F, camera_matrix_data );
    cv::Mat rvec, tvec, inliers;
    // 求解pnp
    cv::solvePnPRansac( pts_obj, pts_img, cameraMatrix, cv::Mat(), rvec, tvec, false, 100, 0.5, 100, inliers );

//    cout<<"inliers: "<<inliers.rows<<endl;
//    cout<<"R="<<rvec<<endl;
//    cout<<"t="<<tvec<<endl;

    // 画出inliers匹配
    vector< cv::DMatch > matchesShow;
    for (size_t i=0; i<inliers.rows; i++)
    {
        matchesShow.push_back( filteredGoodMatches[inliers.ptr<int>(i)[0]] );
    }
    cv::drawMatches( rgb1, kp1, rgb2, kp2, matchesShow, imgMatches );
    cv::imshow( "inlier matches", imgMatches );
    cv::imwrite( "./data/inliers.png", imgMatches );
    cv::waitKey( 0 );

    cv::Mat R;
    cv::Rodrigues(rvec, R );
//    cout << "Rotation Matrix: " << R << endl;
//    cout << "Translation Vector: " << tvec << endl;
//    cout << "Inverse K: " << cameraMatrix.inv() << endl;
    cv::Mat inverseCameraMatrix = cameraMatrix.inv();

    // prepare for the epiplor search
    // pose
    Eigen::Isometry3d thisToOtherEigen = cvMat2Eigen(rvec, tvec);
    Sophus::SE3Group<double> thisToOther(thisToOtherEigen.matrix());
    Sim3 result(thisToOther.unit_quaternion(), thisToOther.translation());
    result.setScale(1.0);
    cout << result.matrix() << endl;
    Eigen::Matrix3f K;
    cv::cv2eigen(cameraMatrix, K);
    FramePose framePose1(K, result);

    // reference image data
    cv::Mat gray1, gray2;
    cvtColor(rgb1, gray1, CV_RGB2GRAY);
    cvtColor(rgb2, gray2, CV_RGB2GRAY);
    cv::Mat referenceFrameImage;
    cv::Mat keyFrameImage;
    gray1.convertTo(keyFrameImage, CV_32F);
    gray2.convertTo(referenceFrameImage, CV_32F);
    int w = referenceFrameImage.cols;
    int h = referenceFrameImage.rows;
    float* refImgPtr = (float*)referenceFrameImage.data;
    DepthMap depthMap1(w, h, K, keyFrameImage);
    float epxn = 2.0;
    float epyn = 2.0;

    // depth search range
    float maxInverseDepth = 1.0/minDepth;
    float minInverseDepth = 1.0/maxDepth;
    //float trueDepth = depth1.ptr<ushort>( int(pointKeyframe.y) )[ int(pointKeyframe.x) ]/ (float)C.scale;
    float trueDepth = (minInverseDepth + maxInverseDepth)/2;
    float priorInverseDepth = 1.0/trueDepth;
    //cout <<  "minDepth " << minDepth << ", maxDepth " << maxDepth << endl;
    //cout <<  "maxInverseDepth " << maxInverseDepth << ", minInverseDepth " << minInverseDepth << endl;

    for (size_t i=0; i<inliers.rows; i++)
    {
        // point in keyframe
        cv::Point2f pointKeyframe = kp1[matchesShow[i].queryIdx].pt;
        cv::Point2f pointGoundTruth = kp2[matchesShow[i].trainIdx].pt;
        float u_p = pointKeyframe.x;
        float v_p = pointKeyframe.y;
        float estimatedInverseDepth;
        float estimatedVar;
        float estimatedEplLen;

        cout << "Keyframe point: " << pointKeyframe.x << ", " << pointKeyframe.y << endl;
        depthMap1.doLineStereo(
                    u_p, v_p, epxn, epyn,
                    minInverseDepth, priorInverseDepth, maxInverseDepth,
                    framePose1, refImgPtr,
                    estimatedInverseDepth, estimatedVar, estimatedEplLen);
        cout << "Reference frame point: " << pointGoundTruth.x << ", " << pointGoundTruth.y << endl;
        float trueDepth = depth1.ptr<ushort>( int(pointKeyframe.y) )[ int(pointKeyframe.x) ]/ (float)C.scale;
        cout << "Estimated Depth: " << 1.0/estimatedInverseDepth << ", Ground Truth: " << trueDepth << endl;
        cout << endl;
    }
}


