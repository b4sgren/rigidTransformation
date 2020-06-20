#include <gtest/gtest.h>
#include <Eigen/Core>
#include <iostream>
#include <random>

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

Eigen::Vector2d randVec2d(double min, double max)
{
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<double> dist(min, max);

    Eigen::Vector2d vec;
    vec << dist(generator), dist(generator);
    return vec;
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

// TEST(ResultOfGroupMultiplication, ReturnsNewMemberOfGroup)
// {
//     for(int i{0}; i != 100; ++i)
//     {
//         SO2<double> R1 = SO2<double>::random();
//         SO2<double> R2 = SO2<double>::random();

//         SO2<double> R3 = R1 * R2;
//         double ang1{getAngle(R1.R())}, ang2{getAngle(R2.R())};
//         double ang3{ang1 + ang2};
//         ang3 = wrap(ang3);
//         SO2<double> R3_true = SO2<double>::fromAngle(ang3);

//         EXPECT_TRUE(R3_true.R().isApprox(R3.R(), 1e-8));
//     }
// }

// TEST(InverseOfSO2Object, ReturnsIdentityWhenMultipliedByInverse)
// {
//     for(int i{0}; i != 100; i++)
//     {
//         SO2<double> R{SO2<double>::random()};
//         SO2<double> R_inv{R.inv()};

//         SO2<double> I = R * R_inv;

//         EXPECT_TRUE(I.R().isApprox(Eigen::Matrix2d::Identity()));
//     }
// }

// TEST(InverseInPlace, InvertsObject)
// {
//     for(int i{0}; i != 100; ++i)
//     {
//         SO2<double> R{SO2<double>::random()};
//         Eigen::Matrix2d inv = R.R().transpose();
//         R.selfInv();

//         EXPECT_TRUE(inv.isApprox(R.R()));
//     }
// }

// TEST(ActiveRotation, RotatedVector)
// {
//     for(int i{0}; i != 100; ++i)
//     {
//         SO2<double> R{SO2<double>::random()};
//         Eigen::Vector2d vec{randVec2d(-10.0, 10.0)};
        
//         Eigen::Vector2d res{R.rota(vec)};

//         double ang{getAngle(R.R())};
//         double ct{cos(ang)}, st{sin(ang)};
//         Eigen::Vector2d res_true{Eigen::Vector2d::Zero()};
//         res_true(0) = ct * vec(0) - st * vec(1);
//         res_true(1) = st * vec(0) + ct * vec(1);

//         EXPECT_TRUE(res_true.isApprox(res));
//     }
// }

// TEST(PassiveRotation, RotatedVector)
// {
//     for(int i{0}; i != 100; ++i)
//     {
//         SO2<double> R{SO2<double>::random()};
//         Eigen::Vector2d vec{randVec2d(-10.0, 10.0)};
        
//         Eigen::Vector2d res{R.rotp(vec)};

//         double ang{getAngle(R.R())};
//         double ct{cos(ang)}, st{sin(ang)};
//         Eigen::Vector2d res_true{Eigen::Vector2d::Zero()};
//         res_true(0) = ct * vec(0) + st * vec(1);
//         res_true(1) = -st * vec(0) + ct * vec(1);

//         EXPECT_TRUE(res_true.isApprox(res));
//     }
// }

// TEST(HatOperator, GivenAnAngleReturnSkewSymmetricMatrix)
// {
//     for(int i{0}; i != 100; ++i)
//     {
//         SO2<double> R{SO2<double>::random()};
//         double angle{getAngle(R.R())};

//         Eigen::Matrix2d r{SO2<double>::hat(angle)};
//         Eigen::Matrix2d r_true;
//         r_true << 0.0, -angle, angle, 0.0;

//         EXPECT_TRUE(r_true.isApprox(r));
//     }
// }

// TEST(VeeOperator, GivenSkewSymmetricMatrixReturnAngle)
// {
//     for(int i{0}; i != 100; ++i)
//     {
//         SO2<double> R{SO2<double>::random()};
//         double angle{getAngle(R.R())};

//         Eigen::Matrix2d tmp{SO2<double>::hat(angle)};
//         double ang{SO2<double>::vee(tmp)};

//         EXPECT_TRUE(ang==angle);
//     }
// }

// TEST(SkewMethod, Value_ReturnSkewSymmetricMatrix)
// {
//     for(int i{0}; i !=100; ++i)
//     {
//         double j{i};

//     }
// }

// TEST(Ajoint, DISABLED_GivenSO2Element)
// {
//     //Do once Exp is implemented
// }