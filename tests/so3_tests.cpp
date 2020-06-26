#include <gtest/gtest.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <random>

#include "so3.h"

double getRandomDouble()
{
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<double> dist;

    return dist(generator);
}

Eigen::Vector3d getRandomVector()
{
    Eigen::Vector3d v;
    v << getRandomDouble(), getRandomDouble(), getRandomDouble();
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
        Eigen::Vector3d v{getRandomVector()};
        v /= v.norm();
        double ang = getRandomDouble();

        Eigen::Matrix3d R_true{Eigen::AngleAxisd(ang, v)};

        SO3<double> R{SO3<double>::fromAxisAngle(v * ang)};
        EXPECT_TRUE(R_true.isApprox(R.R()));
    }
}