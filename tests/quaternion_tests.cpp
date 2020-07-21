#include <gtest/gtest.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
// #include <eigen3/unsupported/Eigen/MatrixFunctions>
#include <random>

#include "quaternion.h"
#include "so3.h"
#include "utils.h"

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

TEST(Inverse, QuaternionInverse_QuaternionInverse)
{
    Eigen::Vector4d qi;
    qi << 1.0, 0.0, 0.0, 0.0;
    for(int i{0}; i != 100; ++i)
    {
        Quaternion<double> q{Quaternion<double>::random()};

        Quaternion<double> q_inv{q.inv()};
        Quaternion<double> res{q * q_inv};

        EXPECT_TRUE(qi.isApprox(res.q()));
    }
}

TEST(SelfInverse, Quaternion_QuaternionInverse)
{
    Eigen::Vector4d qi;
    qi << 1.0, 0.0, 0.0, 0.0;
    for(int i{0}; i!= 100; ++i)
    {
        Quaternion<double> q{Quaternion<double>::random()};
        Quaternion<double> q2{q};

        q.selfInv();
        Quaternion<double> res{q * q2};

        EXPECT_TRUE(qi.isApprox(res.q()));
    }
}

TEST(QuaternionMultiply, RandomQuaternions_ReturnsConcatenatedRotation)
{
    for(int i{0}; i != 100; ++i)
    {
        Quaternion<double> q1{Quaternion<double>::random()}, q2{Quaternion<double>::random()};
        Quaternion<double> q3{q1 * q2};

        Eigen::Vector4d q_true;
        q_true << q1.qw() * q2.qw() - q1.qx() * q2.qx() - q1.qy() * q2.qy() - q1.qz() * q2.qz(),
                  q1.qw() * q2.qx() + q1.qx() * q2.qw() + q1.qy() * q2.qz() - q1.qz() * q2.qy(),
                  q1.qw() * q2.qy() - q1.qx() * q2.qz() + q1.qy() * q2.qw() + q1.qz() * q2.qx(),
                  q1.qw() * q2.qz() + q1.qx() * q2.qy() - q1.qy() * q2.qx() + q1.qz() * q2.qw();
        fixQuat(q_true);
        
        EXPECT_TRUE(q_true.isApprox(q3.q())); 
    }
}

TEST(ActiveRotation, QuaternionAndVector_ReturnRotatedVector)
{
    for(int i{0}; i != 100; ++i)
    {
        Quaternion<double> q{Quaternion<double>::random()};
        Eigen::Vector3d v{getRandomVector(-10.0, 10.0)};

        Eigen::Vector3d vp{q.rota(v)};
        Eigen::Vector3d res{q.R() * v};

        EXPECT_TRUE(res.isApprox(vp));
    }
}

TEST(PassiveRotation, QuaternionAndVector_ReturnRotatedVector)
{
    for(int i{0}; i != 100; ++i)
    {
        Quaternion<double> q{Quaternion<double>::random()};
        Eigen::Vector3d v{getRandomVector(-10.0, 10.0)};

        Eigen::Vector3d vp{q.rotp(v)};
        Eigen::Vector3d res{q.inv().R() * v};

        EXPECT_TRUE(res.isApprox(vp));
    }
}

TEST(FromAxisAngle, AxisAngle_ReturnValidQuaternion)
{
    for(int i{0}; i != 100; ++i)
    {
        double ang{randomDouble(0, PI)};
        Eigen::Vector3d vec{getRandomVector(-10.0, 10.0)};
        vec = vec / vec.norm() * ang;

        Quaternion<double> q{Quaternion<double>::fromAxisAngle(vec)};
        Eigen::Vector4d q_true;
        q_true << cos(ang/2.0), vec / ang * sin(ang/2.0);
        fixQuat(q_true);

        EXPECT_TRUE(q_true.isApprox(q.q()));
    }
}

TEST(FromAxisAngleTaylorSeries, AxisAngle_ReturnValidQuaternion)
{
    for(int i{0}; i != 100; ++i)
    {
        double ang{randomDouble(0, 1e-6)};
        Eigen::Vector3d vec{getRandomVector(-10.0, 10.0)};
        vec = vec / vec.norm() * ang;

        Quaternion<double> q{Quaternion<double>::fromAxisAngle(vec)};
        SO3<double> R_true{SO3<double>::fromAxisAngle(vec)};

        EXPECT_TRUE(R_true.R().isApprox(q.R()));
    }
}

// TEST(FromRotationMatrix, RotationMatrix_ReturnValidQuaternion)
// {
//     for(int i{0}; i != 100; ++i)
//     {
//         Quaternion<double> q{Quaternion<double>::random()};
//         Eigen::Matrix3d R{q.R()};

//         Quaternion<double> q2{Quaternion<double>::fromRotationMatrix(R)};

//         EXPECT_TRUE(q == q2);
//     }
// }