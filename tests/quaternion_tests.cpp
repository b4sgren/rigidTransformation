#include <gtest/gtest.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
// #include <eigen3/unsupported/Eigen/MatrixFunctions>
#include <random>

#include "quaternion.h"

//Make a helper file for stuff like skew functions and what not

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

TEST(FromAxisAngle, AxisAngle_ReturnValidQuaternion)
{

}