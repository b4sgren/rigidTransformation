#include <gtest/gtest.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
// #include <eigen3/unsupported/Eigen/MatrixFunctions>
#include <random>

#include "quaternion.h"

TEST(Quaternion, Returns4Vector)
{
    Eigen::Vector4d q_true;
    q_true << 1.0, 0.0, 0.0, 0.0;

    Quaternion<double> q{q_true};

    EXPECT_TRUE(q_true.isApprox(q.q()));
}