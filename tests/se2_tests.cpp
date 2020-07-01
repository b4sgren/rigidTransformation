#include <gtest/gtest.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <eigen3/unsupported/Eigen/MatrixFunctions>
#include <random>

#include "se2.h"

double getRandomDouble(double min, double max)
{
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<double> dist(min, max);

    return dist(generator);
}

Eigen::Vector2d getRandomVector(double min, double max)
{
    Eigen::Vector2d v;
    v << getRandomDouble(min, max), getRandomDouble(min, max);
    return v;
}

TEST(SE2_Element, AskedForMatrix_ReturnsHomogeneousTransformationMatrix)
{
    Eigen::Matrix3d T;
    T << 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0;
    Eigen::Matrix2d R{T.block<2,2>(0,0)};
    Eigen::Vector2d t{T.block<2,1>(0,2)};

    SE2<double> T1{T};
    EXPECT_TRUE(T.isApprox(T1.T()));
    EXPECT_TRUE(R.isApprox(T1.R()));
    EXPECT_TRUE(t.isApprox(T1.t()));
}

TEST(GenerateRandomSE2Element, IfTransformationIsValid_ReturnsTrue)
{
    for(int i{0}; i!=100; ++i)
    {
        SE2<double> T{SE2<double>::random()};
        EXPECT_TRUE(T.isValidTransformation());
    }
}

TEST(FromAngleAndVec, AngleAndTranslationVector_ReturnTransformationMatrix)
{
    for(int i{0}; i!=100; ++i)
    {
        double angle{getRandomDouble(-PI, PI)};
        Eigen::Vector2d t{getRandomVector(-10.0, 10.0)};
        
        SE2<double> T{SE2<double>::fromAngleAndVec(angle, t)};
        double ct{cos(angle)}, st{sin(angle)};
        Eigen::Matrix3d T_true;
        T_true << ct, -st, t(0), st, ct, t(1), 0.0, 0.0, 1.0;

        EXPECT_TRUE(T_true.isApprox(T.T()));
    }
}

TEST(GroupMultiplication, TwoSE2Objects_ReturnConcatenatedSE2Object)
{
    for(int i{0}; i != 100; ++i)
    {
        SE2<double> T1{SE2<double>::random()}, T2{SE2<double>::random()};

        SE2<double> T3{T1 * T2};
        Eigen::Matrix3d T3_true;
        T3_true.block<2,2>(0,0) = T1.R() * T2.R();
        T3_true.block<2,1>(0,2) = T1.t() + T1.R() * T2.t();
        T3_true(2,2) = 1.0;

        EXPECT_TRUE(T3_true.isApprox(T3.T()));
    }
}

TEST(Inverse, SE2Object_ReturnInverse)
{
    for(int i{0}; i != 100; ++i)
    {
        SE2<double> T{SE2<double>::random()};
        SE2<double> T_inv{T.inv()};

        SE2<double> res{T * T_inv};

        EXPECT_TRUE(Eigen::Matrix3d::Identity().isApprox(res.T()));
    }
}

TEST(InverseInPlace, SE2Object_InvertsSelf)
{
    for(int i{0}; i != 100; ++i)
    {
        SE2<double> T{SE2<double>::random()};
        Eigen::Matrix3d T_mat{T.T()};

        T.selfInv();
        Eigen::Matrix3d res{T.T() * T_mat};

        EXPECT_TRUE(Eigen::Matrix3d::Identity().isApprox(res));
    }
}

TEST(GroupActionOnVector, SE2ObjectAndPoint_TranslatedPoint)
{
    for(int i{0}; i !=100; ++i)
    {
        SE2<double> T{SE2<double>::random()};
        Eigen::Vector2d v{getRandomVector(-10.0, 10.0)};
        Eigen::Vector3d vn; // HOmogeneous coordinates
        vn << v(0), v(1), 1.0;

        /*
        Note that you get out what you pass in. If you want a vector in 
        homogeneous coordinates then pass in a vector in homogeneous coordinates
        */
        Eigen::Vector2d vp{T * v};
        Eigen::Vector3d vn_p{T * vn};

        Eigen::Vector2d vp_true{T.R() * v + T.t()};

        EXPECT_TRUE(vp_true.isApprox(vp));
        EXPECT_TRUE(vp_true.isApprox(vn_p.segment<2>(0)));
    }
}

