#include <gtest/gtest.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
// #include <eigen3/unsupported/Eigen/MatrixFunctions>
#include <random>

#include "quaternion.h"
#include "so3.h"

TEST(Quaternion, Returns4Vector)
{
    Eigen::Vector4d q_true;
    q_true << 1.0, 0.0, 0.0, 0.0;

    Quaternion<double> q{q_true};

    EXPECT_TRUE(q_true.isApprox(q.q()));
}

TEST(RandomGeneration, RandomQuaternion_ReturnsValidQuaternion)
{
    for(int i{0}; i != 100; ++i)
    {
        Quaternion<double> q{Quaternion<double>::random()};
        EXPECT_TRUE(q.isValidQuaternion());
    }
}

TEST(GetRotationMatrix, RandomQuaternion_ReturnsCorrectRotationMatrix)
{
    for(int i{0}; i != 100; ++i)
    {
        Quaternion<double> q{Quaternion<double>::random()};
        SO3<double> R_true{SO3<double>::fromQuaternion(q.q())};
        Eigen::Matrix3d R{q.R()};

        EXPECT_TRUE(R_true.R().isApprox(R));
    }
}

// TEST(FromAxisAngle, AxisAngle_ReturnValidQuaternion)
// {
    // for(int i{0}; i != 100; ++i)
// }