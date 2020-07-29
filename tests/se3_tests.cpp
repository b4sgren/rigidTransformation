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

TEST(GetRotation, SE3Element_Returns3x3RotationMatrix)
{
    Eigen::Matrix4d T{Eigen::Matrix4d::Random()};
    SE3<double> Td{T};

    Eigen::Matrix3d R{T.block<3,3>(0,0)};
    EXPECT_TRUE(R.isApprox(Td.R()));
}

TEST(GetTranslation, SE3Element_Returns3x1TranslationVector)
{
    Eigen::Matrix4d T{Eigen::Matrix4d::Random()};
    SE3<double> Td{T};

    Eigen::Vector3d t{T.block<3,1>(0,3)};
    EXPECT_TRUE(t.isApprox(Td.t()));
}