#include <gtest/gtest.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <eigen3/unsupported/Eigen/MatrixFunctions>
#include <random>

#include "se2.h"

TEST(SE2_Element, AskedForMatrix_ReturnsHomogeneousTransformationMatrix)
{
    Eigen::Matrix3d T;
    T << 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0;
    Eigen::Matrix2d R{T.block<2,2>(0,0)};
    Eigen::Vector2d t{T.block<2,1>(0,2)};

    SE2<double> T1{T};
    EXPECT_TRUE(T.isApprox(T1.T()));
    EXPECT_TRUE(R.isApprox(T1.R()));
    EXPECT_TRUE(t.isApprox(T1.t()));
}

TEST(GenerateRandomSE2Element, IfTransformationIsValid_ReturnsTrue)
{
    for(int i{0}; i!=100; ++i)
    {
        SE2<double> T{SE2<double>::random()};
        EXPECT_TRUE(T.isValidTransformation());
    }
}
