#include <gtest/gtest.h>
#include <Eigen/Core>
#include <iostream>

#include "so2.h"

//The SO2 class is templated but I mostly care that it works with doubles and ceres jet

TEST(AskForTheRotationMatrix, ReturnsTheRotationMatrix)
{
    Eigen::Matrix2d R_true;
    R_true << 1, 0, 0, 1;

    SO2<double> R(R_true);
    
    EXPECT_TRUE(R_true.isApprox(R.R()));
}

TEST(AskForRandomMatrix, ReturnsMatrixWithDeterminantOne)
{
    for(int i{0}; i != 100; ++i)
    {
        SO2<double> R = SO2<double>::random();
        EXPECT_TRUE(R.isValidRotation());
    }
}

TEST(AskForMatrixFromAngle, ReturnsRotationMatrix)
{
    for(int i{0}; i != 100; ++i)
    {
        SO2<double> R = SO2<double>::random();
        double ang = atan2(R.R()(1,0), R.R()(0,0));
        SO2<double> R2 = SO2<double>::fromAngle(ang);
        EXPECT_TRUE(R.R().isApprox(R2.R()));
    }
}