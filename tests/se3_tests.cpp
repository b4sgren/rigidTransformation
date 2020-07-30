#include <gtest/gtest.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <eigen3/unsupported/Eigen/MatrixFunctions>
#include <random>

#include "se3.h"

double getRandomDouble(double min, double max)
{
    static std::random_device rd;
    static std::mt19937 generator(rd());
    std::uniform_real_distribution<double> dist(min, max);

    return dist(generator);
}

Eigen::Vector3d getRandomVector(double min, double max)
{
    Eigen::Vector3d v;
    v << getRandomDouble(min, max), getRandomDouble(min, max), getRandomDouble(min, max);
    return v;
}

TEST(Constructor, SE3Element_Returns4by4Matrix)
{
    Eigen::Matrix4d T{Eigen::Matrix4d::Random()};
    SE3<double> Td{T};

    EXPECT_TRUE(T.isApprox(Td.T()));
}

TEST(GetRotation, SE3Element_Returns3x3RotationMatrix)
{
    Eigen::Matrix4d T{Eigen::Matrix4d::Random()};
    SE3<double> Td{T};

    Eigen::Matrix3d R{T.block<3,3>(0,0)};
    EXPECT_TRUE(R.isApprox(Td.R()));
}

TEST(GetTranslation, SE3Element_Returns3x1TranslationVector)
{
    Eigen::Matrix4d T{Eigen::Matrix4d::Random()};
    SE3<double> Td{T};

    Eigen::Vector3d t{T.block<3,1>(0,3)};
    EXPECT_TRUE(t.isApprox(Td.t()));
}

TEST(RandomGeneration, RandomSE3Element_VerifyThatElementIsValidTransformation)
{
    for(int i{0}; i != 100; ++i)
    {
        SE3<double> T{SE3<double>::random()};
        EXPECT_TRUE(T.isValidTransformation());
    }
}

TEST(FromRotationMatrixAndVector, RotationMatrixAndVector_ValidTransformationMatrix)
{
    for(int i{0}; i != 0; ++i)
    {
        double ang{getRandomDouble(0, PI)};
        Eigen::Vector3d v{getRandomVector(-10.0, 10.0)};
        v /= v.norm();

        Eigen::Matrix3d R(Eigen::AngleAxisd(ang, v));
        Eigen::Vector3d t{getRandomVector(-10.0, 10.0)};

        SE3<double> T{R, t};

        EXPECT_TRUE(T.isValidTransformation());
    }
}

TEST(FromAxisAngleAndVector, AxisAngle_ReturnValidTransformationMatrix)
{
    for(int i{0}; i != 100; ++i)
    {
        double ang{getRandomDouble(0, PI)};
        Eigen::Vector3d v{getRandomVector(-10.0, 10.0)};
        v /= v.norm();
        Eigen::Vector3d t{getRandomVector(-10.0, 10.0)};

        SE3<double> T{SE3<double>::fromAxisAngleAndVector(v * ang, t)};

        Eigen::Matrix3d R(Eigen::AngleAxisd(ang, v));
        SE3<double> T_true{R, t};

        EXPECT_TRUE(T_true == T);
        EXPECT_TRUE(T.isValidTransformation());
    }
}

TEST(FromAxisAngleAndVector, AxisAngle_ReturnValidTransformationMatrixUsingTaylorSeries)
{
    for(int i{0}; i != 100; ++i)
    {
        double ang{getRandomDouble(0, 1e-6)};
        Eigen::Vector3d v{getRandomVector(-10.0, 10.0)};
        v /= v.norm();
        Eigen::Vector3d t{getRandomVector(-10.0, 10.0)};

        SE3<double> T{SE3<double>::fromAxisAngleAndVector(v * ang, t)};

        Eigen::Matrix3d R(Eigen::AngleAxisd(ang, v));
        SE3<double> T_true{R, t};

        EXPECT_TRUE(T_true == T);
        EXPECT_TRUE(T.isValidTransformation());
    }
}

TEST(FromAxisAngleAndVector, EigenAngleAxid_ReturnValidTransformationMatrix)
{
    for(int i{0}; i != 100; ++i)
    {
        double ang{getRandomDouble(0, 1e-6)};
        Eigen::Vector3d v{getRandomVector(-10.0, 10.0)};
        v /= v.norm();
        Eigen::Vector3d t{getRandomVector(-10.0, 10.0)};

        SE3<double> T{SE3<double>::fromAxisAngleAndVector(Eigen::AngleAxisd(ang, v), t)};
        SE3<double> T_true{SE3<double>::fromAxisAngleAndVector(v * ang, t)};

        EXPECT_TRUE(T_true == T);
        EXPECT_TRUE(T.isValidTransformation());
    }
}

TEST(FromRPYAnglesAndVector, RPYAnglesVecAndVector_ReturnValidTransformationMatrix)
{
    for(int i{0}; i != 100; ++i)
    {
        Eigen::Vector3d rpy{getRandomVector(-PI, PI)};
        Eigen::Vector3d t{getRandomVector(-10.0, 10.0)};

        SE3<double> T{SE3<double>::fromRPYAndVector(rpy, t)};

        Eigen::Matrix3d Rx(Eigen::AngleAxisd(rpy(0), Eigen::Vector3d::UnitX()));
        Eigen::Matrix3d Ry(Eigen::AngleAxisd(rpy(1), Eigen::Vector3d::UnitY()));
        Eigen::Matrix3d Rz(Eigen::AngleAxisd(rpy(2), Eigen::Vector3d::UnitZ()));
        Eigen::Matrix3d R{Rz * Ry * Rx};
        SE3<double> T_true{R, t};

        EXPECT_TRUE(T_true == T);
        EXPECT_TRUE(T.isValidTransformation());
    }
}

TEST(FromRPYAnglesAndVector, RPYAnglesAndVector_ReturnValidTransformationMatrix)
{
    for(int i{0}; i != 100; ++i)
    {
        Eigen::Vector3d rpy{getRandomVector(-PI, PI)};
        Eigen::Vector3d t{getRandomVector(-10.0, 10.0)};

        SE3<double> T{SE3<double>::fromRPYAndVector(rpy(0), rpy(1), rpy(2), t)};

        Eigen::Matrix3d Rx(Eigen::AngleAxisd(rpy(0), Eigen::Vector3d::UnitX()));
        Eigen::Matrix3d Ry(Eigen::AngleAxisd(rpy(1), Eigen::Vector3d::UnitY()));
        Eigen::Matrix3d Rz(Eigen::AngleAxisd(rpy(2), Eigen::Vector3d::UnitZ()));
        Eigen::Matrix3d R{Rz * Ry * Rx};
        SE3<double> T_true{R, t};

        EXPECT_TRUE(T_true == T);
        EXPECT_TRUE(T.isValidTransformation());
    }
}

TEST(FromQuaternion, Quaternion_ReturnValidTransformationMatrix)
{
    for(int i{0}; i != 100; ++i)
    {
        Eigen::Vector3d v{getRandomVector(-10.0, 10.0)};
        v /= v.norm();
        double ang{getRandomDouble(0, PI)};
        Eigen::Vector3d t{getRandomVector(-10.0, 10.0)};

        Eigen::Vector4d q;
        double st = sin(ang/2);
        // q << cos(ang/2), st * v(0), st * v(1), st * v(2);
        q << cos(ang/2), st * v;

        SE3<double> T{SE3<double>::fromQuaternionAndVector(q, t)};
        
        Eigen::Matrix3d R{Eigen::AngleAxisd(ang, v)};
        SE3<double> T_true{R, t};

        EXPECT_TRUE(T.isValidTransformation());
        EXPECT_TRUE(T_true == T);
    }
}

TEST(FromQuaternion, EigenQuaternion_ReturnValidTransformationMatrix)
{
    for(int i{0}; i != 100; ++i)
    {
        Eigen::Vector3d v{getRandomVector(-10.0, 10.0)};
        v /= v.norm();
        double ang{getRandomDouble(0, PI)};
        Eigen::Vector3d t{getRandomVector(-10.0, 10.0)};

        
        Eigen::Quaterniond q{Eigen::AngleAxisd(ang, v)};
        Eigen::Matrix3d R{Eigen::AngleAxisd(ang, v)};
        SE3<double> T{SE3<double>::fromQuaternionAndVector(q, t)};
        SE3<double> T_true{R, t};

        EXPECT_TRUE(T.isValidTransformation());
        EXPECT_TRUE(T_true == T);
    }
}

TEST(FromPointer, PointerInColumnMajorOrder_ReturnValidTransformation)
{
    for(int i{0}; i != 100; ++i)
    {
        SE3<double> T_true{SE3<double>::random()};
        double *p{T_true.data()};

        SE3<double> T(p);
        EXPECT_TRUE(T_true == T);
        EXPECT_TRUE(T.isValidTransformation());
    }
}

TEST(GroupMultiplication, 2SE3Elements_ReturnNewSE3Element)
{
    for(int i{0}; i != 100; ++i)
    {
        SE3<double> T1{SE3<double>::random()}, T2{SE3<double>::random()};

        SE3<double> T3{T1 * T2};
        Eigen::Matrix4d T3_true{T1.T() * T2.T()};

        EXPECT_TRUE(T3_true.isApprox(T3.T()));
    }
}

TEST(Inverse, SE3Element_ReturnInverseElement)
{
    for(int i{0}; i != 100; ++i)
    {
        SE3<double> T{SE3<double>::random()};
        SE3<double> T_inv{T.inv()};
        SE3<double> res{T * T_inv};

        EXPECT_TRUE(Eigen::Matrix4d::Identity().isApprox(res.T()));
    }
}

TEST(Inverse, SE3Element_DoesInverseInPlace)
{
    for(int i{0}; i != 100; ++i)
    {
        SE3<double> T{SE3<double>::random()};
        Eigen::Matrix4d T_inv{T.T().inverse()};
        T.selfInv();

        EXPECT_TRUE(T_inv.isApprox(T.T()));
    }
}