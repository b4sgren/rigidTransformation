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
    {}
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

// TEST(Quaternion, Returns4Vector)
// {
//     Eigen::Vector4d q_true;
//     q_true << 1.0, 0.0, 0.0, 0.0;

//     Quaternion<double> q{q_true};

//     EXPECT_TRUE(q_true.isApprox(q.q()));
// }

// TEST(RandomGeneration, RandomQuaternion_ReturnsValidQuaternion)
// {
//     for(int i{0}; i != 100; ++i)
//     {
//         Quaternion<double> q{Quaternion<double>::random()};
//         EXPECT_TRUE(q.isValidQuaternion());
//     }
// }

// TEST(GetRotationMatrix, RandomQuaternion_ReturnsCorrectRotationMatrix)
// {
//     for(int i{0}; i != 100; ++i)
//     {
//         Quaternion<double> q{Quaternion<double>::random()};
//         SO3<double> R_true{SO3<double>::fromQuaternion(q.q())};
//         Eigen::Matrix3d R{q.R()};

//         EXPECT_TRUE(R_true.R().isApprox(R));
//     }
// }

// TEST(Inverse, QuaternionInverse_QuaternionInverse)
// {
//     Eigen::Vector4d qi;
//     qi << 1.0, 0.0, 0.0, 0.0;
//     for(int i{0}; i != 100; ++i)
//     {
//         Quaternion<double> q{Quaternion<double>::random()};

//         Quaternion<double> q_inv{q.inv()};
//         Quaternion<double> res{q * q_inv};

//         EXPECT_TRUE(qi.isApprox(res.q()));
//     }
// }

// TEST(SelfInverse, Quaternion_QuaternionInverse)
// {
//     Eigen::Vector4d qi;
//     qi << 1.0, 0.0, 0.0, 0.0;
//     for(int i{0}; i!= 100; ++i)
//     {
//         Quaternion<double> q{Quaternion<double>::random()};
//         Quaternion<double> q2{q};

//         q.selfInv();
//         Quaternion<double> res{q * q2};

//         EXPECT_TRUE(qi.isApprox(res.q()));
//     }
// }

// TEST(QuaternionMultiply, RandomQuaternions_ReturnsConcatenatedRotation)
// {
//     for(int i{0}; i != 100; ++i)
//     {
//         Quaternion<double> q1{Quaternion<double>::random()}, q2{Quaternion<double>::random()};
//         Quaternion<double> q3{q1 * q2};

//         Eigen::Vector4d q_true;
//         q_true << q1.qw() * q2.qw() - q1.qx() * q2.qx() - q1.qy() * q2.qy() - q1.qz() * q2.qz(),
//                   q1.qw() * q2.qx() + q1.qx() * q2.qw() + q1.qy() * q2.qz() - q1.qz() * q2.qy(),
//                   q1.qw() * q2.qy() - q1.qx() * q2.qz() + q1.qy() * q2.qw() + q1.qz() * q2.qx(),
//                   q1.qw() * q2.qz() + q1.qx() * q2.qy() - q1.qy() * q2.qx() + q1.qz() * q2.qw();
//         fixQuat(q_true);

//         EXPECT_TRUE(q_true.isApprox(q3.q()));
//     }
// }

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

// TEST(FromAxisAngle, AxisAngle_ReturnValidQuaternion)
// {
//     for(int i{0}; i != 100; ++i)
//     {
//         double ang{randomDouble(0, PI)};
//         Eigen::Vector3d vec{getRandomVector(-10.0, 10.0)};
//         vec = vec / vec.norm() * ang;

//         Quaternion<double> q{Quaternion<double>::fromAxisAngle(vec)};
//         Eigen::Vector4d q_true;
//         q_true << cos(ang/2.0), vec / ang * sin(ang/2.0);
//         fixQuat(q_true);

//         EXPECT_TRUE(q_true.isApprox(q.q()));
//         EXPECT_TRUE(q.isValidQuaternion());
//     }
// }

// TEST(FromAxisAngleTaylorSeries, AxisAngle_ReturnValidQuaternion)
// {
//     for(int i{0}; i != 100; ++i)
//     {
//         double ang{randomDouble(0, 1e-6)};
//         Eigen::Vector3d vec{getRandomVector(-10.0, 10.0)};
//         vec = vec / vec.norm() * ang;

//         Quaternion<double> q{Quaternion<double>::fromAxisAngle(vec)};
//         SO3<double> R_true{SO3<double>::fromAxisAngle(vec)};

//         EXPECT_TRUE(R_true.R().isApprox(q.R()));
//         EXPECT_TRUE(q.isValidQuaternion());
//     }
// }

// TEST(FromRotationMatrix, RotationMatrix_ReturnValidQuaternion)
// {
//     for(int i{0}; i != 100; ++i)
//     {
//         Quaternion<double> q{Quaternion<double>::random()};
//         Eigen::Matrix3d R{q.R()};

//         Quaternion<double> q2{Quaternion<double>::fromRotationMatrix(R)};

//         EXPECT_TRUE(q == q2);
//         EXPECT_TRUE(q2.isValidQuaternion());
//     }
// }

// TEST(FromAxisAngleEigen, EigenAngleAxis_ReturnQuaternion)
// {
//     for(int i{0}; i != 100; ++i)
//     {
//         double ang{randomDouble(0, PI)};
//         Eigen::Vector3d vec{getRandomVector(-10.0, 10.0)};
//         vec /= vec.norm();
//         Eigen::AngleAxisd v{ang, vec};
//         vec *= ang;

//         Quaternion<double> q{Quaternion<double>::fromAxisAngle(v)};
//         Quaternion<double> q2{Quaternion<double>::fromAxisAngle(vec)};

//         EXPECT_TRUE(q == q2);
//         EXPECT_TRUE(q.isValidQuaternion());
//     }
// }

// TEST(FromRPYAngles, RPYEulerAngles_ReturnQuaternion)
// {
//     for(int i{0}; i != 100; ++i)
//     {
//         Eigen::Vector3d rpy{getRandomVector(-PI, PI)};

//         Quaternion<double> q{Quaternion<double>::fromRPY(rpy)};

//         SO3<double> R{SO3<double>::fromRPY(rpy)};
//         Quaternion<double> q_true{Quaternion<double>::fromRotationMatrix(R.R())};

//         EXPECT_TRUE(q_true == q);
//         EXPECT_TRUE(q.isValidQuaternion());
//     }
// }

// TEST(FromPointer, Pointer_ReturnValidQuaternion)
// {
//     for(int i{0}; i != 100; ++i)
//     {
//         Quaternion<double> q{Quaternion<double>::random()};
//         double * data{q.data()};
//         Quaternion<double> q2{data};

//         EXPECT_TRUE(q == q2);
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
