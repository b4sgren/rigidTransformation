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