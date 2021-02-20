#include <gtest/gtest.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
// #include <eigen3/unsupported/Eigen/MatrixFunctions>
#include <random>

#include "quaternion.h"
#include "so3.h"
#include "utils.h"

namespace rt = rigidTransform;
using Quatd = rt::Quaternion<double>;

void fixQuat(Eigen::Vector4d &q)
{
    if(q(0) < 0.0)
        q *= -1;
}

double randomDouble(double min, double max)
{
    static std::random_device rd;
    static std::mt19937 generator(rd());
    static std::uniform_real_distribution<double> dist(min, max);

    return dist(generator);
}

Eigen::Vector3d getRandomVector(double min, double max)
{
    Eigen::Vector3d v;
    v << randomDouble(min, max), randomDouble(min, max), randomDouble(min, max);

    return v;
}

class Quat_Fixture : public ::testing::Test
{
public:
    Quat_Fixture()
    {
        for(int i{0}; i != 100; ++i)
            transforms_.push_back(Quatd::random());
    }

    std::vector<Quatd> transforms_;
};

TEST_F(Quat_Fixture, DefaultInitialization)
{
    Quatd q{};
    Eigen::Vector4d q_true(1, 0, 0, 0);
    EXPECT_TRUE(q_true.isApprox(q.q()));
}

TEST_F(Quat_Fixture, InitializeFromPointer)
{
    double data[]{1, 0, 0, 0};
    Quatd q{data};
    Eigen::Map<Eigen::Vector4d> q_true(data);
    EXPECT_TRUE(q_true.isApprox(q.q()));
}

TEST_F(Quat_Fixture, InitializeFromVector)
{
    Eigen::Vector4d q_true;
    q_true << 1, 0, 0, 0;
    Quatd q{q_true};
    EXPECT_TRUE(q_true.isApprox(q.q()));
}

TEST_F(Quat_Fixture, InitializeFromRPY)
{
    for(int i{0}; i != 100; ++i)
    {
        double r{randomDouble(-rt::PI, rt::PI)};
        double p{randomDouble(-rt::PI, rt::PI)};
        double y{randomDouble(-rt::PI, rt::PI)};
        Quatd q{r, p, y};
        rt::SO3<double> R{r, p, y};

        EXPECT_TRUE(R.R().isApprox(q.R().transpose()));
    }
}

TEST_F(Quat_Fixture, InitializeFromRotationMatrix)
{
    for(int i{0}; i != 100; ++i)
    {
        rt::SO3<double> R{rt::SO3<double>::random()};
        Quatd q{R.R()};
        EXPECT_TRUE(R.R().isApprox(q.R().transpose()));
    }
}

TEST_F(Quat_Fixture, InitializeFromAxisAngle)
{
    for(int i{0}; i != 100; ++i)
    {
        double theta{randomDouble(-rt::PI, rt::PI)};
        Eigen::Vector3d vec{getRandomVector(-10, 10)};
        vec = vec / vec.norm() * theta;

        Quatd q{vec};
        rt::SO3<double> R{vec};

        EXPECT_TRUE(R.R().isApprox(q.R().transpose()));
    }
}

TEST_F(Quat_Fixture, InitializeFromQuaternion)
{
    Eigen::Vector4d vec;
    vec << sqrt(2), 0, sqrt(2), 0;
    Quatd q{vec};
    Quatd q2{q};
    EXPECT_TRUE(q2.q().isApprox(q.q()));
}

TEST_F(Quat_Fixture, AssignmentOperator)
{
    Eigen::Vector4d vec;
    vec << sqrt(2), 0, sqrt(2), 0;
    Quatd q{vec};
    Quatd q2 = q;
    EXPECT_TRUE(q2.q().isApprox(q.q()));

}

TEST_F(Quat_Fixture, RandomInitialization)
{
    for(auto q : transforms_)
        EXPECT_FLOAT_EQ(1.0, q.norm());
}

TEST_F(Quat_Fixture, IdentityInitialization)
{
    Quatd q{Quatd::Identity()};
    Eigen::Vector4d I;
    I << 1, 0, 0, 0;
    EXPECT_TRUE(I.isApprox(q.q()));
}

TEST_F(Quat_Fixture, GroupMultiplication)
{
    for(Quatd q : transforms_)
    {
        Quatd q2{Quatd::random()};

        Quatd q3{q*q2};
        Eigen::Vector4d q3_true{Eigen::Vector4d::Zero()};
        q3_true(0) = q.qw()*q2.qw() - q.qx()*q2.qx() - q.qy()*q2.qy() - q.qz()*q2.qz();
        q3_true(1) = q.qw()*q2.qx() + q.qx()*q2.qw() + q.qy()*q2.qz() - q.qz()*q2.qy();
        q3_true(2) = q.qw()*q2.qy() - q.qx()*q2.qz() + q.qy()*q2.qw() + q.qz()*q2.qx();
        q3_true(3) = q.qw()*q2.qz() + q.qx()*q2.qy() - q.qy()*q2.qx() + q.qz()*q2.qw();

        if(q3_true(0) < 0)
            q3_true *= -1;

        EXPECT_TRUE(q3_true.isApprox(q3.q()));
    }
}

TEST_F(Quat_Fixture, OrderOfMultiplication)
{
    for(int i{0}; i != 100; ++i)
    {
        double ang1{randomDouble(-rt::PI, rt::PI)}, ang2{randomDouble(-rt::PI, rt::PI)};
        Eigen::Vector3d v{0, 0, 1};

        Eigen::Vector3d v1 = v * ang1;
        Quatd q_1_from_orig(v1);
        Eigen::Matrix3d R_1_from_origin(Eigen::AngleAxisd(ang1, Eigen::Vector3d::UnitZ()));

        Eigen::Vector3d v2 = v * ang2;
        Quatd q_2_from_1(v2);
        Eigen::Matrix3d R_2_from_1(Eigen::AngleAxisd(ang2, Eigen::Vector3d::UnitZ()));

        // Note the opposite order of composition!!
        Eigen::Matrix3d resR(R_2_from_1 * R_1_from_origin);
        Quatd resq(q_1_from_orig * q_2_from_1);

        Quatd resqR(resR);
        EXPECT_TRUE(resq.q().isApprox(resqR.q()));
    }
}

TEST_F(Quat_Fixture, QuaternionInverse)
{
    for(Quatd q : transforms_)
    {
        Quatd q_inv(q.inverse());
        Quatd res(q * q_inv);
        Quatd I(Quatd::Identity());

        EXPECT_TRUE(I.q().isApprox(res.q()));
    }
}

TEST_F(Quat_Fixture, InverseInPlace)
{
    for(Quatd q: transforms_)
    {
        Quatd q_orig(q);
        q.inverse_();

        Quatd res(q * q_orig);
        Quatd I(Quatd::Identity());

        EXPECT_TRUE(I.q().isApprox(res.q()));
    }
}

TEST_F(Quat_Fixture, ActiveRotationOfAVector)
{
    for(Quatd q : transforms_)
    {
        Eigen::Vector3d v(getRandomVector(-10, 10));
        Eigen::Vector3d vp(q.rota(v));
        Eigen::Vector3d vp_true(q.R().transpose() * v);

        EXPECT_TRUE(vp_true.isApprox(vp));
    }
}

TEST_F(Quat_Fixture, PassiveRotationOfAVector)
{
    for(Quatd q : transforms_)
    {
        Eigen::Vector3d v(getRandomVector(-10, 10));
        Eigen::Vector3d vp(q.rotp(v));
        Eigen::Vector3d vp_true(q.R() * v);

        EXPECT_TRUE(vp_true.isApprox(vp));

    }
}

// TEST(ActiveRotation, QuaternionAndVector_ReturnRotatedVector)
// {
//     for(int i{0}; i != 100; ++i)
//     {
//         Quaternion<double> q{Quaternion<double>::random()};
//         Eigen::Vector3d v{getRandomVector(-10.0, 10.0)};

//         Eigen::Vector3d vp{q.rota(v)};
//         Eigen::Vector3d vp2{q*v};
//         Eigen::Vector3d res{q.R() * v};

//         EXPECT_TRUE(res.isApprox(vp));
//         EXPECT_TRUE(res.isApprox(vp2));
//     }
// }

// TEST(PassiveRotation, QuaternionAndVector_ReturnRotatedVector)
// {
//     for(int i{0}; i != 100; ++i)
//     {
//         Quaternion<double> q{Quaternion<double>::random()};
//         Eigen::Vector3d v{getRandomVector(-10.0, 10.0)};

//         Eigen::Vector3d vp{q.rotp(v)};
//         Eigen::Vector3d res{q.inv().R() * v};

//         EXPECT_TRUE(res.isApprox(vp));
//     }
// }

// TEST(VeeOperator, PureQuaternion_ReturnTheVectorPortion)
// {
//     for(int i{0}; i != 100; ++i)
//     {
//         Quaternion<double> q{Quaternion<double>::random()};

//         Eigen::Vector3d v1{Quaternion<double>::vee(q.q())}, v2;
//         v2 << q.qx(), q.qy(), q.qz();

//         EXPECT_TRUE(v1.isApprox(v2));
//     }
// }

// TEST(HatOperator, 3Vector_ReturnPureQuaternion)
// {
//     for(int i{0}; i != 100; ++i)
//     {
//         Eigen::Vector3d v{getRandomVector(-PI, PI)};

//         Eigen::Vector4d q1{Quaternion<double>::hat(v)}, q2;
//         q2 << 0.0, v;

//         EXPECT_TRUE(q1.isApprox(q2));
//     }
// }

// TEST(QuaternionExponential, 3Vector_ReturnQuaternion) //No need to test Taylor Series b/c done in fromAxisAngle
// {
//     for(int i{0}; i != 100; ++i)
//     {
//         Eigen::Vector3d vec{getRandomVector(-PI, PI)};

//         Quaternion<double> q{Quaternion<double>::Exp(vec)};
//         SO3<double> R{SO3<double>::Exp(vec)};
//         Quaternion<double> q_true{Quaternion<double>::fromRotationMatrix(R.R())};

//         // EXPECT_EQ(q_true, q); //Doesn't like this
//         EXPECT_TRUE(q_true == q);
//     }
// }

// TEST(QuaternionLogarithm, Quaternion_Return3Vector)
// {
//     for(int i{0}; i != 100; ++i)
//     {
//         Quaternion<double> q{Quaternion<double>::random()};

//         Eigen::Vector3d log_q1{q.Log()};
//         Eigen::Vector3d log_q2{Quaternion<double>::Log(q)};

//         Quaternion<double> q1{Quaternion<double>::Exp(log_q1)};
//         Quaternion<double> q2{Quaternion<double>::Exp(log_q2)};

//         EXPECT_TRUE(q == q1);
//         EXPECT_TRUE (q == q2);
//     }
// }

// TEST(QuaternionLogarithTaylorSeries, Quaternion_Return3Vector)
// {
//     for(int i{0}; i != 100; ++i)
//     {
//         double ang{randomDouble(0, 1e-6)};
//         Eigen::Vector3d vec{getRandomVector(-10.0, 10.0)};
//         vec = vec / vec.norm() * ang;

//         Quaternion<double> q{Quaternion<double>::fromAxisAngle(vec)};

//         Eigen::Vector3d log_q{q.Log()};
//         Quaternion<double> q2{Quaternion<double>::Exp(log_q)};

//         EXPECT_TRUE(q == q2);
//     }
// }

// TEST(BoxPlus, QuaternionAnd3Vector_ReturnsNewQuaternion)
// {
//     for(int i{0}; i != 100; ++i)
//     {
//         Quaternion<double> q{Quaternion<double>::random()};
//         SO3<double> Rq{q.R()};
//         Eigen::Vector3d v{getRandomVector(-PI, PI)};

//         SO3<double> R{Rq.boxplus(v)};
//         Quaternion<double>q_true{Quaternion<double>::fromRotationMatrix(R.R())};
//         Quaternion<double> q2{q.boxplus(v)};

//         EXPECT_TRUE(q_true == q2);
//     }
// }

// TEST(BoxMinus, TwoQuaternions_ReturnDifferenceAs3Vector)
// {
//     for(int i{0}; i != 100; ++i)
//     {
//         Quaternion<double> q1{Quaternion<double>::random()}, q2{Quaternion<double>::random()};

//         Eigen::Vector3d w{q1.boxminus(q2)};
//         Quaternion<double> res{q2.boxplus(w)};

//         EXPECT_TRUE(q1 == res);
//     }
// }

// TEST(Normalize, Quaternion_QuaternionUnitNorm)
// {
//     for(int i{0}; i != 10; ++i)
//     {
//         Quaternion<double> q{Quaternion<double>::random()};
//         q.normalize();

//         EXPECT_TRUE(q.isValidQuaternion());
//     }
// }

// TEST(Identity, AskedForIdentity_ReturnsIdentity)
// {
//     Eigen::Vector4d I;
//     I << 1.0, 0.0, 0.0, 0.0;
//     Quaternion<double> q{Quaternion<double>::Identity()};

//     EXPECT_TRUE(I.isApprox(q.q()));
// }
