#include <gtest/gtest.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <eigen3/unsupported/Eigen/MatrixFunctions>
#include <random>

#include "se3.h"

TEST(Constructor, SE3Element_Returns4by4Matrix)
{
    Eigen::Matrix4d T{Eigen::Matrix4d::Random()};
    SE3<double> Td{T};

    EXPECT_TRUE(T.isApprox(Td.T()));
}