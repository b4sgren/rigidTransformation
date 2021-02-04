#include <gtest/gtest.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <eigen3/unsupported/Eigen/MatrixFunctions>
#include <random>

#include "so3.h"
#include "utils.h"

namespace rt = rigidTransform;

double getRandomDouble(double min, double max)
{
    static std::random_device rd;
    static std::mt19937 generator(rd());
    static std::uniform_real_distribution<double> dist(min, max);

    return dist(generator);
}

Eigen::Vector3d getRandomVector(double min, double max)
{
    Eigen::Vector3d v;
    v << getRandomDouble(min, max), getRandomDouble(min, max), getRandomDouble(min, max);
    return v;
}

class SO3_Fixture : public ::testing::Test
{
public:
    SO3_Fixture()
    {
        for(int i{0}; i != 100; ++i)
        {
            transforms_.push_back(rt::SO3<double>::random());
        }
    }

    ~SO3_Fixture()
    {
        transforms_.clear();
    }

protected:
    std::vector<rt::SO3<double>> transforms_;
};

TEST_F(SO3_Fixture, TestDefaultInitialization)
{
    rt::SO3<double> R{};
    Eigen::Matrix3d I{Eigen::Matrix3d::Identity()};

    EXPECT_TRUE(I.isApprox(R.R()));
}

TEST_F(SO3_Fixture, TestPointerInitialization)
{
    double vals[]{1, 2, 3, 4, 5, 6, 7, 8, 9};
    rt::SO3<double> R{vals};
    Eigen::Matrix3d M;
    M << 1, 2, 3, 4, 5, 6, 7, 8, 9;
    M.transposeInPlace();

    EXPECT_TRUE(M.isApprox(R.R()));
}

TEST_F(SO3_Fixture, TestInitializationFromMatrix)
{
    Eigen::Matrix3d M{Eigen::Matrix3d::Random()};
    rt::SO3<double> R{M};

    EXPECT_TRUE(M.isApprox(R.R()));
}

TEST_F(SO3_Fixture, TestFromRPYAngles)
{
    for(int i{0}; i != 100; ++i)
    {
        double roll{getRandomDouble(-rt::PI, rt::PI)};
        double pitch{getRandomDouble(-rt::PI, rt::PI)};
        double yaw{getRandomDouble(-rt::PI, rt::PI)};
        rt::SO3<double> R{roll, pitch, yaw};

        Eigen::AngleAxisd rollAngle(roll, Eigen::Vector3d::UnitX());
        Eigen::AngleAxisd pitchAngle(pitch, Eigen::Vector3d::UnitY());
        Eigen::AngleAxisd yawAngle(yaw, Eigen::Vector3d::UnitZ());

        Eigen::Matrix3d R_true = yawAngle.matrix() * pitchAngle.matrix() * rollAngle.matrix();

        EXPECT_TRUE(R_true.isApprox(R.R()));
    }
}

TEST_F(SO3_Fixture, TestFromAxisAngle)
{
    for(int i{0}; i != 100; ++i)
    {
        double phi{getRandomDouble(0, rt::PI)};
        Eigen::Vector3d vec{Eigen::Vector3d::Random()};
        Eigen::Vector3d v = vec / vec.norm() * phi;

        rt::SO3<double> R{v};
        Eigen::Matrix3d R_true = Eigen::AngleAxisd(phi, vec/vec.norm()).matrix();

        EXPECT_TRUE(R_true.isApprox(R.R()));
    }
}

TEST_F(SO3_Fixture, InitializeFromSO3)
{
    Eigen::Matrix3d M{Eigen::Matrix3d::Random()};
    rt::SO3<double> R{M};
    rt::SO3<double> R2{R};

    EXPECT_TRUE(R.R().isApprox(R2.R()));
}

TEST_F(SO3_Fixture, TestAssignmentOperator)
{
    Eigen::Matrix3d M{Eigen::Matrix3d::Random()};
    rt::SO3<double> R{M};
    rt::SO3<double> R2 = R;

    EXPECT_TRUE(R.R().isApprox(R2.R()));
}

TEST_F(SO3_Fixture, TestRandomInitialization)
{
    for(int i{0}; i != 100; ++i)
    {
        rt::SO3<double> R{rt::SO3<double>::random()};
        EXPECT_FLOAT_EQ(1.0, R.R().determinant());
    }
}

TEST_F(SO3_Fixture, TestIdentityFunction)
{
    rt::SO3<double> R{rt::SO3<double>::Identity()};
    Eigen::Matrix3d I{Eigen::Matrix3d::Identity()};
    EXPECT_TRUE(I.isApprox(R.R()));
}

TEST_F(SO3_Fixture, TestGroupMultiplication)
{
    for(auto R : transforms_)
    {
        rt::SO3<double> R2{rt::SO3<double>::random()};
        rt::SO3<double> R3{R * R2};

        Eigen::Matrix3d R3_true = R.R() * R2.R();

        EXPECT_TRUE(R3_true.isApprox(R3.R()));
    }
}

TEST_F(SO3_Fixture, TestInverse)
{
    Eigen::Matrix3d I{Eigen::Matrix3d::Identity()};
    for(auto R : transforms_)
    {
        rt::SO3<double> R_inv{R.inverse()};
        rt::SO3<double> res{R * R_inv};
        EXPECT_TRUE(I.isApprox(res.R()));
    }
}

TEST_F(SO3_Fixture, TestInPlaceInverse)
{
    Eigen::Matrix3d I{Eigen::Matrix3d::Identity()};
    for(auto R : transforms_)
    {
        rt::SO3<double> R_orig{R};
        R.inverse_();
        rt::SO3<double> res{R * R_orig};
        EXPECT_TRUE(I.isApprox(res.R()));
    }

}

// TEST(Inverse, AskedForInverse_InverseTimesOriginalGivesIdentity)
// {
//     for(int i{0}; i != 100; ++i)
//     {
//         SO3<double> R{SO3<double>::random()};
//         SO3<double> R_inv{R.inv()};

//         EXPECT_TRUE(Eigen::Matrix3d::Identity().isApprox(R_inv.R() * R.R()));
//     }
// }

// TEST(SelfInverse, SO3Element_InvertsInPlace)
// {
//     for(int i{0}; i!=100; ++i)
//     {
//         SO3<double> R{SO3<double>::random()};
//         Eigen::Matrix3d R_inv{R.R()};
//         R.selfInv();

//         EXPECT_TRUE(Eigen::Matrix3d::Identity().isApprox(R_inv * R.R()));
//     }
// }

// TEST(GroupOperator, TwoSO3ElementsMultiplied_ReturnsValidRotation)
// {
//     for(int i{0}; i != 100; ++i)
//     {
//         SO3<double> R1{SO3<double>::random()};
//         SO3<double> R2{SO3<double>::random()};

//         SO3<double> R3{R1 * R2};

//         Eigen::Matrix3d R_true{R1.R() * R2.R()};

//         EXPECT_TRUE(R_true.isApprox(R3.R()));
//         EXPECT_TRUE(R3.isValidRotation());
//     }
// }

// TEST(ActiveRotation, SO3ElementAnd3Vector_ReturnActivelyRotatedVector)
// {
//     for(int i{0}; i != 100; ++i)
//     {
//         SO3<double> R{SO3<double>::random()};
//         Eigen::Vector3d v{getRandomVector(-10.0, 10.0)};

//         Eigen::Vector3d vp{R.rota(v)};
//         Eigen::Vector3d vp_true{R.R() * v};

//         EXPECT_TRUE(vp_true.isApprox(vp));
//     }
// }

// TEST(PassiveRotation, SO3ElementAnd3Vector_ReturnPassivelyRotatedVector)
// {
//     for(int i{0}; i!=100; ++i)
//     {
//         SO3<double> R{SO3<double>::random()};
//         Eigen::Vector3d v{getRandomVector(-10.0, 10.0)};

//         Eigen::Vector3d vp{R.rotp(v)};
//         Eigen::Vector3d vp_true{R.inv().R() * v};

//         EXPECT_TRUE(vp_true.isApprox(vp));
//     }
// }

// TEST(MatrixLogarithm, SO3Element_ReturnsMatrixLogarithm)
// {
//     for(int i{0}; i!=100; ++i)
//     {
//         SO3<double> R{SO3<double>::random()};

//         Eigen::Matrix3d log_R{R.log()};
//         Eigen::Matrix3d log_R_true{R.R().log()};

//         auto norm{(log_R_true.array() - log_R.array()).matrix().norm()};
//         if(norm >= 1e-8)
//         {
//             std::cout << "Truth\n" << log_R_true << std::endl;
//             std::cout << "Mine\n" << log_R << std::endl;
//             std::cout << "Diff:\t" << (log_R_true.array() - log_R.array()).matrix().norm() << std::endl;
//             Eigen::Matrix3d temp{R.log()};
//             int x{3};
//         }

//         EXPECT_LE(norm, 1e-8);
//     }
// }

// TEST(MatrixLogarithm, SO3Element_ReturnsMatrixLogarithmUsingTaylorSeriesAboutZero)
// {
//     for(int i{0}; i!=100; ++i)
//     {
//         Eigen::Vector3d v{getRandomVector(-10.0, 10.0)};
//         v /= v.norm();
//         double ang{getRandomDouble(0.0, 1e-6)};

//         Eigen::Matrix3d R{Eigen::AngleAxisd(ang, v)};
//         SO3<double> R1{R};

//         Eigen::Matrix3d log_R_true{R.log()};
//         Eigen::Matrix3d log_R{R1.log()};

//         auto norm{(log_R_true.array() - log_R.array()).matrix().norm()};
//         if(!(norm <= 1e-8))
//         {
//             std::cout << "Truth\n" << log_R_true << std::endl;
//             std::cout << "Mine\n" << log_R << std::endl;
//             std::cout << norm << std::endl;
//             int x{3};
//         }

//         EXPECT_LE(norm, 1e-8);
//     }
// }

// TEST(MatrixLogarithm, SO3Element_ReturnsMatrixLogarithmUsingTaylorSeriesAboutPI) //Issues with this one in python also
// {
//     for(int i{0}; i!=100; ++i)
//     {
//         Eigen::Vector3d v{getRandomVector(-10.0, 10.0)};
//         v /= v.norm();
//         double ang{PI - getRandomDouble(0, 1e-6)};

//         SO3<double> R1{SO3<double>::fromAxisAngle(ang * v)};
//         Eigen::Matrix3d R{R1.R()};

//         Eigen::Matrix3d log_R_true{R.log()};
//         Eigen::Matrix3d log_R{R1.log()};

//         auto norm{(log_R_true.array() - log_R.array()).matrix().norm()};
//         if(norm > 1e-8)
//         {
//             std::cout << "\nTruth\n" << log_R_true << std::endl;
//             std::cout << "Mine\n" << log_R << std::endl;
//             std::cout << norm << std::endl;
//             Eigen::Matrix3d temp{R1.log()};
//         }

//         EXPECT_LT(norm, 1e-8);
//     }
// }

// TEST(MatrixLogarithm, GivenSO3Element_Returns3Vector)
// {
//     for(int i{0}; i != 100; ++i)
//     {
//         SO3<double> R{SO3<double>::random()};

//         Eigen::Vector3d w1{R.Log()}, w2{SO3<double>::Log(R)}, w3{SO3<double>::Log(R.R())};
//         Eigen::Matrix3d log_R{R.R().log()};
//         Eigen::Vector3d w{SO3<double>::vee(log_R)};

//         auto norm1{(w - w1).norm()}, norm2{(w - w2).norm()}, norm3{(w - w3).norm()};

//         EXPECT_LT(norm1, 1e-8);
//         EXPECT_LT(norm2, 1e-8);
//         EXPECT_LT(norm3, 1e-8);
//     }
// }

// TEST(VeeOperator, SkewSymmetrixMatrix_Return3Vector)
// {
//     for(int i{0}; i != 100; ++i)
//     {
//         Eigen::Vector3d w_true{getRandomVector(-10.0, 10.0)};
//         Eigen::Matrix3d log_R{skew3(w_true)};
//         Eigen::Vector3d w{SO3<double>::vee(log_R)};

//         EXPECT_TRUE(w_true.isApprox(w));
//     }
// }

// TEST(HatOperator, Vector_ReturnsSkewSymmetricMatrix)
// {
//     for(int i{0}; i!= 100; ++i)
//     {
//         Eigen::Vector3d w{getRandomVector(-10.0, 10.0)};
//         Eigen::Matrix3d log_R{SO3<double>::hat(w)};
//         Eigen::Matrix3d log_R_true;
//         log_R_true << 0.0, -w(2), w(1), w(2), 0.0, -w(0), -w(1), w(0), 0.0;

//         EXPECT_TRUE(log_R_true.isApprox(log_R));
//     }
// }

// TEST(MatrixExponential, SkewSymmetricMatrix_ReturnSO3Element)
// {
//     for(int i{0}; i != 100; ++i)
//     {
//         Eigen::Vector3d w{getRandomVector(-PI, PI)};
//         Eigen::Matrix3d log_R{SO3<double>::hat(w)};

//         SO3<double> R{SO3<double>::exp(log_R)};
//         Eigen::Matrix3d R_true{log_R.exp()};

//         EXPECT_TRUE(R_true.isApprox(R.R()));
//     }
// }

// TEST(MatrixExponential, SkewSymmetricMatrix_ReturnsSO3ElementUsingTaylorSeries)
// {
//     for(int i{0}; i != 100; ++i)
//     {
//         Eigen::Vector3d w{getRandomVector(-10.0, 10.0)};
//         double ang{getRandomDouble(0, 1e-6)};
//         w = w / w.norm() * ang;
//         Eigen::Matrix3d log_R{SO3<double>::hat(w)};

//         SO3<double> R{SO3<double>::exp(log_R)};
//         Eigen::Matrix3d R_true{log_R.exp()};

//         EXPECT_TRUE(R_true.isApprox(R.R()));
//     }
// }

// TEST(MatrixExponential, Vector_ReturnSO3Element)
// {
//     for(int i{0}; i != 100; ++i)
//     {
//         Eigen::Vector3d w{getRandomVector(-PI, PI)};

//         SO3<double> R{SO3<double>::Exp(w)};
//         Eigen::Matrix3d log_R{SO3<double>::hat(w)};
//         Eigen::Matrix3d R_true{log_R.exp()};

//         EXPECT_TRUE(R_true.isApprox(R.R()));
//     }
// }

// TEST(Adjoint, SO3ElementAnd3Vector_ComposeWithAdjoint)
// {
//     for(int i{0}; i != 100; ++i)
//     {
//         SO3<double> R{SO3<double>::random()};
//         Eigen::Vector3d w{getRandomVector(-PI, PI)};

//         SO3<double> R2{R * SO3<double>::Exp(w)};
//         SO3<double> R3{SO3<double>::Exp(R.Adj() * w) * R};

//         EXPECT_TRUE(R2 == R3);
//     }
// }

// TEST(BoxPlus, SO3AndVector_ReturnsConcatenationOfTheTwo)
// {
//     for(int i{0}; i!=100; ++i)
//     {
//         SO3<double> R{SO3<double>::random()};
//         Eigen::Vector3d v{getRandomVector(-10.0, 10.0)};
//         v /= v.norm();
//         double ang{getRandomDouble(0.0, PI)};

//         SO3<double> R2{R.boxplus(v*ang)};
//         SO3<double> R2_true{R * SO3<double>::fromAxisAngle(v*ang)};

//         EXPECT_EQ(R2_true, R2);
//     }
// }

// TEST(BoxMinus, SO3Elements_ReturnDifferenceBetweenTheTwo)
// {
//     for(int i{0}; i != 100; ++i)
//     {
//         SO3<double> R1{SO3<double>::random()}, R2{SO3<double>::random()};

//         Eigen::Vector3d w{R1.boxminus(R2)};
//         SO3<double> R3{R2.boxplus(w)};

//         double norm{(R1.R() - R3.R()).norm()};

//         EXPECT_LE(norm, 1e-8);
//     }
// }

// TEST(Normalize, GivenSO3ElementWithDetNotEq1_NormalizeElement)
// {
//     SO3<double> R1{SO3<double>::random()}, R2{SO3<double>::random()};

//     // while(R1.isValidRotation()) //Really slow to not be valid
//         // R1 *= R2;

//     R1.normalize();

//     EXPECT_TRUE(R1.isValidRotation());
// }

// TEST(Identity, AskedForIdentity_ReturnsIdentity)
// {
//     Eigen::Matrix3d I{Eigen::Matrix3d::Identity()};
//     SO3<double> R{SO3<double>::Identity()};

//     EXPECT_TRUE(I.isApprox(R.R()));
// }
