#include <gtest/gtest.h>
#include <Eigen/Core>
#include <iostream>

#include "so2.h"

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