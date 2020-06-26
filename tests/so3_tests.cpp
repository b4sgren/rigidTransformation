#include <gtest/gtest.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <random>

#include "so3.h"

double getRandomDouble(double min, double max)
{
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<double> dist(min, max);

    return dist(generator);
}

Eigen::Vector3d getRandomVector(double min, double max)
{
    Eigen::Vector3d v;
    v << getRandomDouble(min, max), getRandomDouble(min, max), getRandomDouble(min, max);
    return v;
}

TEST(GivenSO3Element, Return3By3Matrix)
{
   Eigen::Matrix3d R_true;
   R_true << 1, 0, 0, 0, 1, 0, 0, 0, 1; 

   SO3<double> R{R_true};

   EXPECT_TRUE(R_true.isApprox(R.R()));
}

TEST(RandomGeneration, AskedForRandomRotationMatrix_ReturnsValidRotationMatrix)
{
    for(int i{0}; i != 100; ++i)
    {
        SO3<double> R{SO3<double>::random()};
        EXPECT_TRUE(R.isValidRotation());
    }
}

TEST(FromAxisAngle, SuppliedAxisAngleVector_ReturnValidRotationMatrix)
{
    for(int i{0}; i != 100; ++i)
    {
        Eigen::Vector3d v{getRandomVector(-10.0, 10.0)};
        v /= v.norm();
        double ang = getRandomDouble(0.0, PI);

        Eigen::Matrix3d R_true{Eigen::AngleAxisd(ang, v)};

        SO3<double> R{SO3<double>::fromAxisAngle(v * ang)};
        EXPECT_TRUE(R_true.isApprox(R.R()));
        EXPECT_TRUE(R.isValidRotation());
    }
}

TEST(FromAxisAngleTaylorSeries, SuppliedAxisAngleVector_ReturnsValidRotationMatrix)
{
    for(int i{0}; i!=100; ++i)
    {
        Eigen::Vector3d v{getRandomVector(-10.0, 10.0)};
        v /= v.norm();
        double ang{getRandomDouble(0, 1e-6)};

        Eigen::Matrix3d R_true{Eigen::AngleAxisd(ang, v)};
        SO3<double> R{SO3<double>::fromAxisAngle(v * ang)};
        EXPECT_TRUE(R.isValidRotation());
        EXPECT_TRUE(R_true.isApprox(R.R()));
    }
}

TEST(FromAxisAngleEigen, SuppliedAxisAngleVector_ReturnsValidRotation)
{
    for(int i{0}; i != 100; ++i)
    {
        Eigen::Vector3d v{getRandomVector(-10.0, 10.0)};
        v /= v.norm();
        double ang = getRandomDouble(0.0, PI);

        Eigen::Matrix3d R_true{Eigen::AngleAxisd(ang, v)};

        SO3<double> R{SO3<double>::fromAxisAngle(Eigen::AngleAxisd(ang, v))};
        EXPECT_TRUE(R_true.isApprox(R.R()));
        EXPECT_TRUE(R.isValidRotation());
    }
}