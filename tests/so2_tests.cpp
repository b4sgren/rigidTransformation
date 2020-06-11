#include <gtest/gtest.h>
#include <Eigen/Core>
#include <iostream>

#include "so2.h"

//The SO2 class is templated but I mostly care that it works with doubles and ceres jet
double getAngle(const Eigen::Matrix2d &mat)
{
    double y{mat(1,0)}, x{mat(0,0)};
    return atan2(y, x);
}

double wrap(double ang)
{
    const double pi{3.14159265};
    ang = ang > pi ? ang - 2 * pi : ang;
    ang = ang <= -pi ? ang + 2 * pi : ang;
    return ang;
}

TEST(AskForTheRotationMatrix, ReturnsTheRotationMatrix)
{
    Eigen::Matrix2d R_true;
    R_true << 1, 0, 0, 1;

    SO2<double> R(R_true);
    
    EXPECT_TRUE(R_true.isApprox(R.R()));
}

TEST(AskForRandomMatrix, ReturnsMatrixWithDeterminantOne)
{
    for(int i{0}; i != 100; ++i)
    {
        SO2<double> R = SO2<double>::random();
        EXPECT_TRUE(R.isValidRotation());
    }
}

TEST(AskForMatrixFromAngle, ReturnsRotationMatrix)
{
    for(int i{0}; i != 100; ++i)
    {
        SO2<double> R = SO2<double>::random();
        double ang = atan2(R.R()(1,0), R.R()(0,0));
        SO2<double> R2 = SO2<double>::fromAngle(ang);
        EXPECT_TRUE(R.R().isApprox(R2.R()));
    }
}

TEST(ResultOfGroupMultiplication, ReturnsNewMemberOfGroup)
{
    for(int i{0}; i != 100; ++i)
    {
        SO2<double> R1 = SO2<double>::random();
        SO2<double> R2 = SO2<double>::random();

        SO2<double> R3 = R1 * R2;
        double ang1{getAngle(R1.R())}, ang2{getAngle(R2.R())};
        double ang3{ang1 + ang2};
        ang3 = wrap(ang3);
        SO2<double> R3_true = SO2<double>::fromAngle(ang3);

        EXPECT_TRUE(R3_true.R().isApprox(R3.R(), 1e-8));
    }
}

TEST(InverseOfSO2Object, ReturnsIdentityWhenMultipliedByInverse)
{
    for(int i{0}; i != 100; i++)
    {
        SO2<double> R{SO2<double>::random()};
        SO2<double> R_inv{R.inv()};

        SO2<double> I = R * R_inv;

        EXPECT_TRUE(I.R().isApprox(Eigen::Matrix2d::Identity()));
    }
}