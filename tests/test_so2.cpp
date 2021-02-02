#include <gtest/gtest.h>
#include <Eigen/Core>
#include <eigen3/unsupported/Eigen/MatrixFunctions>
#include <iostream>
#include <random>

#include "so2.h"

namespace rt = rigidTransform;

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

double getRandomDouble(double min, double max)
{
    static std::random_device rd;
    static std::mt19937 generator(rd());
    static std::uniform_real_distribution<double> dist(min, max);
    return dist(generator);
}

Eigen::Vector2d randVec2d(double min, double max)
{
    Eigen::Vector2d vec;
    vec << getRandomDouble(min, max), getRandomDouble(min, max);
    return vec;
}

class SO2_Fixture : public ::testing::Test
{
public:
    SO2_Fixture()
    {
        for(int i{0}; i != 100; ++i)
        {
            transforms_.push_back(rt::SO2<double>::random());
        }
    }

    ~SO2_Fixture()
    {
        transforms_.clear();
    }

protected:
    std::vector<rt::SO2<double>> transforms_;
};

TEST_F(SO2_Fixture, TestDefaultInitialization)
{
    rt::SO2<double> R = rt::SO2<double>();
    Eigen::Matrix2d R_default{Eigen::Matrix2d::Identity()};

    EXPECT_TRUE(R_default.isApprox(R.R()));
}

TEST_F(SO2_Fixture, TestPointerInitialization)
{
    double theta = rt::PI/6;
    double ct{cos(theta)}, st{sin(theta)};
    double data[]{ct, st, -st, ct};
    rt::SO2<double> R(data);
    Eigen::Matrix2d R_true;
    R_true << ct, -st, st, ct;

    EXPECT_TRUE(R_true.isApprox(R.R()));
}

TEST_F(SO2_Fixture, TestEigenMatrixInitializtion)
{
    double theta{rt::PI/6};
    double ct{cos(theta)}, st{sin(theta)};
    Eigen::Matrix2d R_true;
    R_true << ct, -st, st, ct;
    rt::SO2<double> R{R_true};

    EXPECT_TRUE(R_true.isApprox(R.R()));
}

TEST_F(SO2_Fixture, TestAngleInitialization)
{
    double theta{getRandomDouble(-rt::PI, rt::PI)};
    rt::SO2<double> R(theta);
    double ct{cos(theta)}, st{sin(theta)};
    Eigen::Matrix2d R_true;
    R_true << ct, -st, st, ct;

    EXPECT_TRUE(R_true.isApprox(R.R()));
}

TEST_F(SO2_Fixture, InitializeFromSO2)
{
    double theta{getRandomDouble(-rt::PI, rt::PI)};
    rt::SO2<double> R(theta);
    rt::SO2<double> R2(R);

    EXPECT_TRUE(R.R().isApprox(R2.R()));
}

TEST_F(SO2_Fixture, InitialzeWithAssingmentOperator)
{
    double theta{getRandomDouble(-rt::PI, rt::PI)};
    rt::SO2<double> R(theta);
    rt::SO2<double> R2 = R;

    EXPECT_TRUE(R.R().isApprox(R2.R()));
}

// TEST_F(SO2_Fixture, DISABLED_RandomInitialization)
TEST_F(SO2_Fixture, RandomInitialization)
{
    // Sometimes it still fails randomly
    for(auto R : transforms_)
        EXPECT_FLOAT_EQ(1, R.det());
}

TEST_F(SO2_Fixture, GroupMultiplication)
{
    for(auto R1 : transforms_)
    {
        rt::SO2<double> R2{rt::SO2<double>::random()};
        rt::SO2<double> R3{R1 * R2};
        double ang1{getAngle(R1.R())}, ang2{getAngle(R2.R())};
        double ang3{ang1+ang2};
        ang3 = wrap(ang3);
        rt::SO2<double> R3_true{ang3};

        EXPECT_TRUE(R3_true.R().isApprox(R3.R(), 1e-8));
    }
}

TEST_F(SO2_Fixture, InverseOfGroupElement)
{
    for(auto R : transforms_)
    {
        rt::SO2<double> R_inv{R.inverse()};
        Eigen::Matrix2d res{(R * R_inv).R()};
        Eigen::Matrix2d I{Eigen::Matrix2d::Identity()};

        EXPECT_TRUE(I.isApprox(res));
    }
}

TEST_F(SO2_Fixture, InverseInPlace)
{
    for(auto R : transforms_)
    {
        Eigen::Matrix2d R_orig{R.R()};
        R.inverse_();
        Eigen::Matrix2d I{Eigen::Matrix2d::Identity()};
        Eigen::Matrix2d res{R.R() * R_orig};

        EXPECT_TRUE(I.isApprox(res));
    }
}

TEST_F(SO2_Fixture, ActiveRotationOfAVector)
{
    for(auto R : transforms_)
    {
        Eigen::Vector2d v{randVec2d(-10, 10)};
        Eigen::Vector2d vp{R.rota(v)};

        double ang{getAngle(R.R())};
        double ct{cos(ang)}, st{sin(ang)};
        Eigen::Vector2d vp_true;
        vp_true(0) = ct * v(0) - st * v(1);
        vp_true(1) = st * v(0) + ct * v(1);

        EXPECT_TRUE(vp_true.isApprox(vp));
    }
}

TEST_F(SO2_Fixture, PassiveRotationOfAVector)
{
    for(auto R : transforms_)
    {
        Eigen::Vector2d v{randVec2d(-10, 10)};
        Eigen::Vector2d vp{R.rotp(v)};

        double ang{getAngle(R.R())};
        double ct{cos(ang)}, st{sin(ang)};
        Eigen::Vector2d vp_true;
        vp_true(0) = ct * v(0) + st * v(1);
        vp_true(1) = -st * v(0) + ct * v(1);

        EXPECT_TRUE(vp_true.isApprox(vp));
    }
}

TEST_F(SO2_Fixture, IdentityFunction)
{
    rt::SO2<double> R{rt::SO2<double>::identity()};
    Eigen::Matrix2d I{Eigen::Matrix2d::Identity()};

    EXPECT_TRUE(I.isApprox(R.R()));
}

TEST_F(SO2_Fixture, LogarithmicMap)
{
    for(auto R : transforms_)
    {
        double logR{R.Log()};
        double ang{getAngle(R.R())};

        EXPECT_FLOAT_EQ(ang, logR);
    }
}

TEST_F(SO2_Fixture, ExponentialMap)
{
    for(auto R : transforms_)
    {
        double logR{R.Log()};
        rt::SO2<double> R2{rt::SO2<double>::Exp(logR)};

        EXPECT_TRUE(R.R().isApprox(R2.R()));
    }
}

TEST_F(SO2_Fixture, AdjointOperator)
{
    for(auto R : transforms_)
    {
        double delta{getRandomDouble(-rt::PI, rt::PI)};

        rt::SO2<double> R1{R * rt::SO2<double>::Exp(delta)};
        rt::SO2<double> R2{rt::SO2<double>::Exp(R.Adj() * delta) * R};

        EXPECT_TRUE(R1.R().isApprox(R2.R()));
    }
}

TEST_F(SO2_Fixture, RightBoxPlus)
{
    for(auto R : transforms_)
    {
        double delta{getRandomDouble(-rt::PI, rt::PI)};

        rt::SO2<double> R2{R.boxplusr(delta)};
        double phi(getAngle(R.R()));
        phi += delta;
        phi = wrap(phi);
        rt::SO2<double> R2_true(phi);

        EXPECT_TRUE(R2_true.R().isApprox(R2.R(), 1e-8));
    }
}

TEST_F(SO2_Fixture, RightBoxMinus)
{
    for(auto R1 : transforms_)
    {
        rt::SO2<double> R2{rt::SO2<double>::random()};
        double delta{R1.boxminusr(R2)};
        rt::SO2<double> R{R2.boxplusr(delta)};

        EXPECT_TRUE(R1.R().isApprox(R.R()));
    }
}

// TEST(BoxPlus, GivenSO2AndDelta_ReturnNewSO2)
// {
//     for(int i{0}; i!=100; ++i)
//     {
//         SO2<double> R{SO2<double>::random()};
//         double delta{getRandomDouble(-PI, PI)};
//         double phi = getAngle(R.R());

//         SO2<double> R2_true{SO2<double>::fromAngle(delta + phi)};
//         SO2<double> R2{R.boxplus(delta)};

//         EXPECT_TRUE(R2 == R2_true);
//     }
// }

// TEST(BoxMinus, Given2SO2Elements_ReturnDelta)
// {
//     for(int i{0}; i!=100; ++i)
//     {
//         SO2<double> R1{SO2<double>::random()};
//         SO2<double> R2{SO2<double>::random()};
//         double phi1{getAngle(R1.R())}, phi2{getAngle(R2.R())};

//         double delta{R1.boxminus(R2)};
//         SO2<double> R{R2.boxplus(delta)};

//         EXPECT_TRUE(R1 == R);
//     }
// }

// TEST(Identity, AskedForIdentity_ReturnsIdentity)
// {
//     Eigen::Matrix2d I{Eigen::Matrix2d::Identity()};
//     SO2<double> R{SO2<double>::Identity()};

//     EXPECT_TRUE(I.isApprox(R.R()));
// }
