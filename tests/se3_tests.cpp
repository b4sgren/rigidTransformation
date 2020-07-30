#include <gtest/gtest.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <eigen3/unsupported/Eigen/MatrixFunctions>
#include <random>

#include "se3.h"

double getRandomDouble(double min, double max)
{
    static std::random_device rd;
    static std::mt19937 generator(rd());
    static std::uniform_real_distribution<double> dist;

    return dist(generator);
}

Eigen::Vector3d getRandomVector(double min, double max)
{
    Eigen::Vector3d v;
    v << getRandomDouble(min, max), getRandomDouble(min, max), getRandomDouble(min, max);
    return v;
}

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

TEST(RandomGeneration, RandomSE3Element_VerifyThatElementIsValidTransformation)
{
    for(int i{0}; i != 100; ++i)
    {
        SE3<double> T{SE3<double>::random()};
        EXPECT_TRUE(T.isValidTransformation());
    }
}

TEST(FromRotationMatrixAndVector, RotationMatrixAndVector_ValidTransformationMatrix)
{
    for(int i{0}; i != 0; ++i)
    {
        double ang{getRandomDouble(0, PI)};
        Eigen::Vector3d v{getRandomVector(-10.0, 10.0)};
        v /= v.norm();

        Eigen::Matrix3d R(Eigen::AngleAxisd(ang, v));
        Eigen::Vector3d t{getRandomVector(-10.0, 10.0)};

        SE3<double> T{R, t};

        EXPECT_TRUE(T.isValidTransformation());
    }
}