#include <gtest/gtest.h>
#include <Eigen/Core>
#include <random>

#include "so3.h"

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