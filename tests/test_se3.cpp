#include <gtest/gtest.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <eigen3/unsupported/Eigen/MatrixFunctions>
#include <random>
#include <vector>

#include "se3.h"
#include "quaternion.h"
#include "so3.h"

namespace rt = rigidTransform;
using SE3d = rt::SE3<double>;
using Quatd = rt::Quaternion<double>;
using Vector7d = Eigen::Matrix<double,7,1>;

double randomDouble(double min, double max)
{
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<double> dist(min, max);

    return dist(generator);
}

Eigen::Vector3d getRandomVector(double min, double max)
{
    Eigen::Vector3d v;
    v << randomDouble(min, max), randomDouble(min, max), randomDouble(min, max);
    return v;
}

template<int n>
bool compareMat(const Eigen::Ref<const Eigen::Matrix<double, n, 1>> &v1, const Eigen::Ref<const Eigen::Matrix<double, n, 1>> &v2)
{
    return v1.isApprox(v2);
}

class SE3_Fixture : public ::testing::Test
{
public:
    SE3_Fixture()
    {
        for(int i{0}; i != 100; ++i)
            transforms_.push_back(SE3d::random());
    }

    ~SE3_Fixture(){}

protected:
    std::vector<SE3d> transforms_;
};

TEST_F(SE3_Fixture, DefaultInitialization)
{
    SE3d T;
    Vector7d I(Vector7d::Zero());
    I(3) = 1.0;
    EXPECT_TRUE(compareMat<7>(I, T.T()));
    EXPECT_TRUE(compareMat<4>(I.tail<4>(), T.q()));
}

TEST_F(SE3_Fixture, PointerInitialization)
{
    double data[]{1,2,3,4,5,6,7};
    SE3d T(data);
    Eigen::Map<Vector7d> T_true(data);
    EXPECT_TRUE(compareMat<7>(T_true, T.T()));
    EXPECT_TRUE(compareMat<4>(T_true.tail<4>(), T.q()));
}

TEST_F(SE3_Fixture, InitializeFromVector)
{
    Vector7d T_true;
    T_true << 1,2,3,4,5,6,7;

    SE3d T(T_true);
    EXPECT_TRUE(compareMat<7>(T_true, T.T()));
    EXPECT_TRUE(compareMat<4>(T_true.tail<4>(), T.q()));
}

TEST_F(SE3_Fixture, InitializeFromRPY)
{
    for(int i{0}; i != 100; ++i)
    {
        double r{randomDouble(-rt::PI, rt::PI)};
        double p{randomDouble(-rt::PI, rt::PI)};
        double y{randomDouble(-rt::PI, rt::PI)};
        Eigen::Vector3d t(getRandomVector(-10, 10));
        SE3d T(r,p,y,t);

        Quatd q{r, p, y};

        EXPECT_TRUE(compareMat<3>(t, T.t()));
        EXPECT_TRUE(compareMat<4>(q.q(), T.q()));
    }
}

TEST_F(SE3_Fixture, FromRotationMatrixAndt)
{
    for(int i{0}; i != 100; ++i)
    {
        Quatd q(Quatd::random());
        Eigen::Vector3d t(getRandomVector(-10, 10));
        SE3d T(q.R().transpose(), t);

        EXPECT_TRUE(compareMat<3>(t, T.t()));
        EXPECT_TRUE(compareMat<4>(q.q(), T.q()));
    }
}

TEST_F(SE3_Fixture, FromQuatAndt)
{
    for(int i{0}; i != 100; ++i)
    {
        Quatd q(Quatd::random());
        Eigen::Vector3d t(getRandomVector(-10, 10));

        SE3d T(q,t);
        EXPECT_TRUE(compareMat<3>(t, T.t()));
        EXPECT_TRUE(compareMat<4>(q.q(), T.q()));
    }
}

TEST_F(SE3_Fixture, FromAxisAngleAndt)
{
    for(int i{0}; i != 100; ++i)
    {
        double theta{randomDouble(-rt::PI, rt::PI)};
        Eigen::Vector3d v(getRandomVector(-10, 10));
        v = v/v.norm() * theta;
        Eigen::Vector3d t(getRandomVector(-10, 10));

        Quatd q(q.fromAxisAngle(v));
        SE3d T(T.fromAxisAngleAndt(v,t));

        EXPECT_TRUE(compareMat<3>(t,T.t()));
        EXPECT_TRUE(compareMat<4>(q.q(), T.q()));
    }
}

TEST_F(SE3_Fixture, FromSE3)
{
    Vector7d T_true;
    T_true << 1, 2, 3, 4, 5, 6, 7;
    SE3d T(T_true);
    SE3d T2(T);

    EXPECT_TRUE(compareMat<7>(T.T(), T2.T()));
}

TEST_F(SE3_Fixture, AssignmentOperator)
{
    Vector7d T_true;
    T_true << 1, 2, 3, 4, 5, 6, 7;
    SE3d T(T_true);
    SE3d T2 = T;

    EXPECT_TRUE(compareMat<7>(T.T(), T2.T()));
}

TEST_F(SE3_Fixture, RandomInitialization)
{
    for(int i{0}; i != 100; ++i)
    {
        SE3d T(SE3d::random());
        EXPECT_FLOAT_EQ(1.0, T.q().norm());
    }
}

TEST_F(SE3_Fixture, GroupMultiplication)
{
    for(SE3d T : transforms_)
    {
        SE3d T2(SE3d::random());
        SE3d T3(T * T2);

        Eigen::Matrix4d TR, T2R, T3R;
        TR << T.R().transpose(), T.t(), 0, 0, 0, 1;
        T2R << T2.R().transpose(), T2.t(), 0, 0, 0, 1;

        T3R = TR * T2R;
        SE3d T3_true(T3R.block<3,3>(0,0), T3R.block<3,1>(0,3));

        EXPECT_TRUE(compareMat<3>(T3_true.t(), T3.t()));
        EXPECT_TRUE(compareMat<4>(T3_true.q(), T3.q()));
    }
}

TEST_F(SE3_Fixture, OrderOfGroupMultiplication) // Check this. It passes but the order of quaternion multiplication isn't backwards...
{
    for(int i{0}; i != 100; ++i)
    {
        Eigen::Vector3d t1(getRandomVector(-10, 10));
        Eigen::Vector3d t2(getRandomVector(-10, 10));
        Eigen::Vector3d vec(getRandomVector(-10, 10));
        vec = vec/vec.norm();
        double ang1(randomDouble(-rt::PI, rt::PI)), ang2(randomDouble(-rt::PI, rt::PI));

        SE3d T_1_from_orig(SE3d::fromAxisAngleAndt(vec*ang1, t1));
        SE3d T_2_from_1(SE3d::fromAxisAngleAndt(vec*ang2, t2));
        SE3d T_2_from_orig(T_2_from_1 * T_1_from_orig);

        Eigen::Matrix3d R_1_from_origin(rt::SO3<double>::fromAxisAngle(vec*ang1).R());
        Eigen::Matrix3d R_2_from_1(rt::SO3<double>::fromAxisAngle(vec*ang2).R());
        Eigen::Matrix4d TR_1_from_orig, TR_2_from_1;
        TR_1_from_orig << R_1_from_origin, t1, 0, 0, 0, 1;
        TR_2_from_1 << R_2_from_1, t2, 0, 0, 0, 1;
        Eigen::Matrix4d TR_2_from_orig = TR_2_from_1 * TR_1_from_orig;

        SE3d res(TR_2_from_orig.block<3,3>(0,0), TR_2_from_orig.block<3,1>(0,3));
        EXPECT_TRUE(compareMat<3>(res.t(), T_2_from_orig.t()));
        EXPECT_TRUE(compareMat<4>(res.q(), T_2_from_orig.q()));
    }
}

TEST_F(SE3_Fixture, Inverse)
{
    for(SE3d T : transforms_)
    {
        SE3d T_inv(T.inverse());
        SE3d T3(T * T_inv);
        Vector7d I(Vector7d::Zero());
        I(3) = 1.0;

        EXPECT_TRUE(compareMat<7>(I, T3.T()));
    }
}

TEST_F(SE3_Fixture, InverseInPlace)
{
    for(SE3d T : transforms_)
    {
        SE3d T_orig(T);
        T.inverse_();
        SE3d res(T * T_orig);
        Vector7d I(Vector7d::Zero());
        I(3) = 1.0;

        EXPECT_TRUE(compareMat<7>(I, res.T()));
    }
}

TEST_F(SE3_Fixture, ActiveTransformation)
{
    for(SE3d T : transforms_)
    {
        Eigen::Vector3d v(getRandomVector(-10, 10));
        Eigen::Vector3d vp(T.transa(v));
        Eigen::Vector3d vp_true(T.t() + T.quat().rota(v));

        EXPECT_TRUE(compareMat<3>(vp_true, vp));
    }
}

TEST_F(SE3_Fixture, PassiveTransformation)
{
    for(SE3d T : transforms_)
    {
        Eigen::Vector3d v(getRandomVector(-10, 10));
        Eigen::Vector3d vp(T.transp(v));
        // Eigen::Vector3d vp_true(T.t() + T.quat().rota(v));
        Eigen::Vector3d res(T.transa(vp));

        EXPECT_TRUE(compareMat<3>(v, res));
    }
}

// TEST(MatrixLogarithm, SE3Element_Returns4by4MatrixInLieAlgebra)
// {
//     for(int i{0}; i != 100; ++i)
//     {
//         SE3<double> T{SE3<double>::random()};

//         Eigen::Matrix4d log_T_true{T.T().log()};
//         Eigen::Matrix4d log_T{T.log()};
//         Eigen::Matrix4d log_T2{SE3<double>::log(T)};
//         Eigen::Matrix4d log_T3{SE3<double>::log(T.T())}; //Can call on not elements of the class

//         double norm{(log_T_true - log_T).norm()};
//         if(norm > 1e-8)
//         {
//             std::cout << "\n\n" << log_T_true << std::endl;
//             std::cout << log_T << std::endl;
//             std::cout << norm << std::endl;
//         }

//         EXPECT_TRUE(log_T2.isApprox(log_T));
//         EXPECT_TRUE(log_T3.isApprox(log_T2));
//         EXPECT_LE(norm, 1e-8);
//     }
// }

// TEST(MatrixLogarithm, SE3Element_ReturnsMatInLieAlgebraUsingTaylorSeriesAbout0)
// {
//     for(int i{0}; i != 100; ++i)
//     {
//         double ang{getRandomDouble(0.0, 1e-6)};
//         Eigen::Vector3d vec{getRandomVector(-10.0, 10.0)};
//         vec  = vec / vec.norm() * ang;
//         Eigen::Vector3d t{getRandomVector(-10.0, 10.0)};

//         SE3<double> T{SE3<double>::fromAxisAngleAndVector(vec, t)};

//         Eigen::Matrix4d log_T_true{T.T().log()};
//         Eigen::Matrix4d log_T{T.log()};

//         double norm{(log_T_true - log_T).norm()};

//         if(norm > 1e-6 || std::isnan(norm))
//         {
//             std::cout << "\n\n" << log_T_true << std::endl;
//             std::cout << log_T << std::endl;
//             std::cout << norm << std::endl;
//             Eigen::Matrix4d temp{T.log()};
//         }

//         EXPECT_LE(norm, 1e-6);
//     }
// }

// TEST(MatrixLogarithm, DISABLED_SE3Element_ReturnsMatInLieAlgebraUsingTaylorSeriesAboutPI)
// {
//     for(int i{0}; i != 100; ++i)
//     {
//         double ang{PI - getRandomDouble(0.0, 1e-6)};
//         Eigen::Vector3d vec{getRandomVector(-10.0, 10.0)};
//         vec  = vec / vec.norm() * ang;
//         Eigen::Vector3d t{getRandomVector(-10.0, 10.0)};

//         SE3<double> T{SE3<double>::fromAxisAngleAndVector(vec, t)};

//         Eigen::Matrix4d log_T_true{T.T().log()};
//         Eigen::Matrix4d log_T{T.log()};

//         double norm{(log_T_true - log_T).norm()};

//         if(norm > 1e-8)
//         {
//             std::cout << "\n\n" << log_T_true << std::endl;
//             std::cout << log_T << std::endl;
//             std::cout << norm << std::endl;
//         }

//         EXPECT_LE(norm, 1e-8); //Haven't worked this out in python either
//     }
// }

// TEST(MatrixExponential, ElementOfSE3LieAlgebra_ReturnSE3Element)
// {
//     for(int i{0}; i != 100; ++i)
//     {
//         Eigen::Vector3d w{getRandomVector(-PI, PI)}, t{getRandomVector(-10.0, 10.0)};
//         Eigen::Matrix4d logT;
//         logT << skew3(w), t, Eigen::RowVector4d::Zero();

//         Eigen::Matrix4d T_true{logT.exp()};
//         SE3<double> T{SE3<double>::exp(logT)};

//         EXPECT_TRUE(T_true.isApprox(T.T()));
//     }
// }

// TEST(MatrixExponential, ElementOfSE3LieAlgebra_ReturnsSE3ElementUsingTaylorSeriesAobut0)
// {
//     for(int i{0}; i != 100; ++i)
//     {
//         double ang{getRandomDouble(0.0, 1e-6)};
//         Eigen::Vector3d w{getRandomVector(-10.0, 10.0)}, t{getRandomVector(-10.0, 10.0)};
//         w = w / w.norm() * ang;

//         Eigen::Matrix4d logT;
//         logT << skew3(w), t, Eigen::RowVector4d::Zero();

//         Eigen::Matrix4d T_true{logT.exp()};
//         SE3<double> T{SE3<double>::exp(logT)};

//         EXPECT_TRUE(T_true.isApprox(T.T()));
//     }
// }

// TEST(HatOperator, SixVector_Returns4x4MatrixOfLieAlgebra)
// {
//     for(int i{0}; i != 100; ++i)
//     {
//         Eigen::Vector3d w{getRandomVector(-PI, PI)}, u{getRandomVector(-10.0, 10.0)};
//         Eigen::Matrix<double, 6, 1> xci;
//         xci << u, w;

//         Eigen::Matrix4d logT{SE3<double>::hat(xci)};
//         Eigen::Matrix4d log_T_true;
//         log_T_true << 0, -w(2), w(1), u(0), w(2), 0, -w(0), u(1), -w(1), w(0), 0, u(2), 0, 0, 0, 0;

//         EXPECT_TRUE(log_T_true.isApprox(logT));
//     }
// }

// TEST(VeeOperator, 4x4Matrix_Returns6Vector)
// {
//     for(int i{0}; i != 100; ++i)
//     {
//         Eigen::Vector3d w{getRandomVector(-PI, PI)}, u{getRandomVector(-10.0, 10.0)};
//         Eigen::Matrix<double,6,1> xci_true;
//         xci_true << u, w;

//         Eigen::Matrix4d logT{SE3<double>::hat(xci_true)};
//         Eigen::Matrix<double,6,1> xci{SE3<double>::vee(logT)};

//         EXPECT_TRUE(xci_true.isApprox(xci));
//     }
// }

// TEST(MatrixLogarithm, ElementOfSE3_ReturnsA6Vector)
// {
//     for(int i{0}; i != 100; ++i)
//     {
//         SE3<double> T{SE3<double>::random()};

//         Eigen::Matrix<double,6,1> xci{T.Log()}, xci_true; //Same 3 calls as other log method
//         Eigen::Matrix4d logT{T.T().log()};

//         xci_true << logT.block<3,1>(0,3), logT(2,1), logT(0,2), logT(1,0);

//         double norm{(xci_true - xci).norm()};

//         EXPECT_LT(norm, 1e-8);
//     }
// }

// TEST(MatrixExponential, 6Vector_ReturnElementOfSE3)
// {
//     for(int i{0}; i != 100; ++i)
//     {
//         Eigen::Vector3d w{getRandomVector(-PI, PI)}, t{getRandomVector(-10.0, 10.0)};
//         Eigen::Matrix<double,6,1> xci;
//         xci << t, w;
//         Eigen::Matrix4d logT;
//         logT << 0, -w(2), w(1), t(0), w(2), 0, -w(0), t(1), -w(1), w(0), 0, t(2), 0, 0, 0, 0;

//         Eigen::Matrix4d T_true{logT.exp()};
//         SE3<double> T{SE3<double>::Exp(xci)};

//         EXPECT_TRUE(T_true.isApprox(T.T()));
//     }
// }

// TEST(BoxPlus, SE3ElementAnd6Vector_ReturnsSE3ElementWhenAdded)
// {
//     for(int i{0}; i != 100; ++i)
//     {
//         SE3<double> T{SE3<double>::random()};
//         Eigen::Vector3d w{getRandomVector(-PI, PI)}, t{getRandomVector(-10.0, 10.0)};
//         Eigen::Matrix<double,6,1> xci;
//         xci << t, w;

//         SE3<double> T_res{T.boxplus(xci)};
//         Eigen::Matrix4d T_true{T.T() * SE3<double>::hat(xci).exp()};
//     }
// }

// TEST(BoxMinus, ElementsOfSE3_6VectorRepresentingTheDifference)
// {
//     for(int i{0}; i != 100; ++i)
//     {
//         SE3<double> T1{SE3<double>::random()}, T2{SE3<double>::random()};

//         Eigen::Matrix<double,6,1>xci{T1.boxminus(T2)};

//         SE3<double> T{T2.boxplus(xci)};

//         if(!(T == T1))
//         {
//             std::cout << "T1:\n" << T1.T() << std::endl;
//             std::cout << "T2:\n" << T.T() << std::endl;
//         }

//         EXPECT_TRUE(T == T1);
//     }
// }

// TEST(Adjoint, ElementOfSE3And6Vector_MultipliedOnLeftEqualsMultipliedOnRight)
// {
//     for(int i{0}; i != 100; ++i)
//     {
//         SE3<double> T1{SE3<double>::random()};
//         Eigen::Vector3d w{getRandomVector(-PI, PI)}, t{getRandomVector(-10.0, 10.0)};
//         Eigen::Matrix<double,6,1> xci;
//         xci << t, w;

//         SE3<double> T2{T1.boxplus(xci)};
//         SE3<double> T3{SE3<double>::Exp(T1.Adj() * xci) * T1};

//         EXPECT_TRUE(T2 == T3);
//     }
// }

// TEST(Identity, AskedForIdentity_ReturnsIdentity)
// {
//     Eigen::Matrix4d I{Eigen::Matrix4d::Identity()};
//     SE3<double> T{SE3<double>::Identity()};

//     EXPECT_TRUE(I.isApprox(T.T()));
// }
