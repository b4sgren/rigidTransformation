#include <gtest/gtest.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <eigen3/unsupported/Eigen/MatrixFunctions>
#include <random>
#include <vector>

#include "se2.h"

namespace rt = rigidTransform;
using SE2d = rt::SE2<double>;

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

bool compareMat(const Eigen::Ref<const Eigen::Matrix3d> &T1, const Eigen::Ref<const Eigen::Matrix3d> &T2)
{
    return T1.isApprox(T2);
}

class SE2_Fixture : public ::testing::Test
{
public:
    SE2_Fixture()
    {
        for(int i{0}; i != 100; ++i)
            transforms_.push_back(SE2d::random());
    }

    ~SE2_Fixture(){}
protected:
    std::vector<SE2d> transforms_;
};

TEST_F(SE2_Fixture, DefaultInitialization)
{
    SE2d T;
    Eigen::Matrix3d I(Eigen::Matrix3d::Identity());
    EXPECT_TRUE(compareMat(I, T.T()));
}

TEST_F(SE2_Fixture, PointerInitialization)
{
    double sqrt2{sqrt(2)};
    double data[]{sqrt2, sqrt2, 0, -sqrt2, sqrt2, 0, 1, 2, 1};
    SE2d T(data);
    Eigen::Matrix3d T_true;
    T_true << sqrt2, -sqrt2, 1, sqrt2, sqrt2, 2, 0, 0, 1;

    EXPECT_TRUE(compareMat(T_true, T.T()));
}

TEST_F(SE2_Fixture, EigenMatrixInitialization)
{
    double sqrt2{sqrt(2)};
    Eigen::Matrix3d T_true;
    T_true << sqrt2, -sqrt2, 1, sqrt2, sqrt2, 2, 0, 0, 1;
    SE2d T(T_true);

    EXPECT_TRUE(compareMat(T_true, T.T()));
}

TEST_F(SE2_Fixture, FromAngleAndt)
{
    double theta{getRandomDouble(-rt::PI, rt::PI)};
    Eigen::Vector2d t(getRandomVector(-10, 10));

    SE2d T1(t, theta);
    SE2d T2(t(0), t(1), theta);

    double ct{cos(theta)}, st{sin(theta)};
    Eigen::Matrix3d T_true;
    T_true << ct, -st, t(0), st, ct, t(1), 0, 0, 1;

    EXPECT_TRUE(compareMat(T_true, T1.T()));
    EXPECT_TRUE(compareMat(T_true, T2.T()));
}

TEST_F(SE2_Fixture, FromRAndt)
{
    double theta{getRandomDouble(-rt::PI, rt::PI)};
    Eigen::Vector2d t(getRandomVector(-10, 10));

    double ct{cos(theta)}, st{sin(theta)};
    Eigen::Matrix2d R;
    R << ct, -st, st, ct;

    SE2d T1(R, t);
    SE2d T2(R, t(0), t(1));

    Eigen::Matrix3d T_true;
    T_true << R, t, 0, 0, 1;

    EXPECT_TRUE(compareMat(T_true, T1.T()));
    EXPECT_TRUE(compareMat(T_true, T2.T()));
}

TEST_F(SE2_Fixture, InitializeFromSE2)
{
    Eigen::Matrix3d T_true{Eigen::Matrix3d::Random()};
    SE2d T(T_true);
    SE2d T2(T);

    EXPECT_TRUE(compareMat(T_true, T2.T()));
}

TEST_F(SE2_Fixture, AssignmentOperator)
{
    Eigen::Matrix3d T_true{Eigen::Matrix3d::Random()};
    SE2d T(T_true);
    SE2d T2 = T;

    EXPECT_TRUE(compareMat(T_true, T2.T()));
}

TEST_F(SE2_Fixture, RandomInitialization)
{
    for(int i{0}; i != 100; ++i)
    {
        SE2d T(SE2d::random());
        double det{T.R().determinant()};

        EXPECT_FLOAT_EQ(1.0, det);
    }
}

TEST_F(SE2_Fixture, GroupMultiplication)
{
    for(SE2d T1 : transforms_)
    {
        SE2d T2(SE2d::random());

        SE2d T3(T1*T2);
        Eigen::Matrix3d T3_true(T1.T() * T2.T());

        EXPECT_TRUE(compareMat(T3_true, T3.T()));
    }
}

TEST_F(SE2_Fixture, InverseOfGroupObject)
{
    for(SE2d T : transforms_)
    {
        SE2d T_inv(T.inverse());
        SE2d res(T * T_inv);
        Eigen::Matrix3d I{Eigen::Matrix3d::Identity()};

        EXPECT_TRUE(compareMat(I, res.T()));
    }
}

TEST_F(SE2_Fixture, InverseInPlace)
{
    for(SE2d T : transforms_)
    {
        SE2d T_copy = T;
        T.inverse_();
        SE2d res(T * T_copy);

        Eigen::Matrix3d I{Eigen::Matrix3d::Identity()};
        EXPECT_TRUE(compareMat(I, res.T()));
    }
}

TEST_F(SE2_Fixture, ActiveTransformation)
{
    for(SE2d T : transforms_)
    {
        Eigen::Vector2d pt(getRandomVector(-10, 10));
        Eigen::Vector2d pt2(T.transa(pt));

        Eigen::Vector3d pth;
        pth << pt, 1;
        Eigen::Vector3d res(T.T() * pth);

        EXPECT_TRUE(res.head<2>().isApprox(pt2));
    }
}

TEST_F(SE2_Fixture, PassiveTransformation)
{
    for(SE2d T : transforms_)
    {
        Eigen::Vector2d pt(getRandomVector(-10, 10));
        Eigen::Vector2d pt2(T.transp(pt));

        Eigen::Vector3d pth;
        pth << pt, 1;
        Eigen::Vector3d res(T.inverse().T() * pth);

        EXPECT_TRUE(res.head<2>().isApprox(pt2));
    }
}


// TEST(ActiveTransformation, SE2ObjectAndPoint_ReturnsTransformedPoint)
// {
//     for(int i{0}; i !=100; ++i)
//     {
//         SE2<double> T{SE2<double>::random()};
//         Eigen::Vector2d v{getRandomVector(-10.0, 10.0)};
//         Eigen::Vector3d vn; // HOmogeneous coordinates
//         vn << v(0), v(1), 1.0;

//         /*
//         Note that you get out what you pass in. If you want a vector in
//         homogeneous coordinates then pass in a vector in homogeneous coordinates
//         */
//         Eigen::Vector2d vp{T.transa(v)};
//         Eigen::Vector3d vn_p{T.transa(vn)};

//         Eigen::Vector2d vp_true{T.R() * v + T.t()};

//         EXPECT_TRUE(vp_true.isApprox(vp));
//         EXPECT_TRUE(vp_true.isApprox(vn_p.segment<2>(0)));
//     }
// }

// TEST(PassiveTransformation, SE2ObjectAndPoint_ReturnsTransformedPoint)
// {
//     for(int i{0}; i !=100; ++i)
//     {
//         SE2<double> T{SE2<double>::random()};
//         Eigen::Vector2d v{getRandomVector(-10.0, 10.0)};
//         Eigen::Vector3d vn; // Homogeneous coordinates
//         vn << v(0), v(1), 1.0;

//         /*
//         Note that you get out what you pass in. If you want a vector in
//         homogeneous coordinates then pass in a vector in homogeneous coordinates
//         */
//         Eigen::Vector2d vp{T.transp(v)};
//         Eigen::Vector3d vn_p{T.transp(vn)};

//         Eigen::Vector2d vp_true{T.inv().R() * v + T.inv().t()};

//         EXPECT_TRUE(vp_true.isApprox(vp));
//         EXPECT_TRUE(vp_true.isApprox(vn_p.segment<2>(0)));
//     }
// }

// TEST(HatOperator, ThreeVector_3by3MatrixOfLieAlgebra)
// {
//     for(int i{0}; i != 100; ++i)
//     {
//         Eigen::Vector2d t{getRandomVector(-10.0, 10.0)};
//         double ang{getRandomDouble(-PI, PI)};
//         Eigen::Vector3d w;
//         w << ang, t(0), t(1);

//         Eigen::Matrix3d log_T{SE2<double>::hat(w)};
//         Eigen::Matrix3d log_T_true;
//         log_T_true << 0, -ang, t(0), ang, 0, t(1), 0, 0, 0;

//         EXPECT_TRUE(log_T_true.isApprox(log_T));
//     }
// }

// TEST(VeeOperator, se2Element_Return3Vector)
// {
//     for(int i{0}; i !=100; ++i)
//     {
//         Eigen::Vector2d t{getRandomVector(-10.0, 10.0)};
//         double ang{getRandomDouble(-PI, PI)};
//         Eigen::Vector3d w_true;
//         w_true << ang, t(0), t(1);

//         Eigen::Matrix3d log_T{SE2<double>::hat(w_true)};
//         Eigen::Vector3d w{SE2<double>::vee(log_T)};

//         EXPECT_TRUE(w_true.isApprox(w));
//     }
// }

// TEST(MatrixLogarithm, SE2Element_ReturnElementOfLieAlgebra)
// {
//     for(int i{0}; i != 100; ++i)
//     {
//         SE2<double> T{SE2<double>::random()};

//         Eigen::Matrix3d log_T{T.log()};
//         Eigen::Matrix3d log_T2{SE2<double>::log(T)};
//         Eigen::Matrix3d log_T3{SE2<double>::log(T.T())};
//         Eigen::Matrix3d log_T_true{T.T().log()};

//         EXPECT_TRUE(log_T_true.isApprox(log_T));
//         EXPECT_TRUE(log_T_true.isApprox(log_T2));
//         EXPECT_TRUE(log_T_true.isApprox(log_T3));
//     }
// }

// TEST(TaylorMatrixLogarithm, SE2Element_ReturnsElementOfLieAlgebra)
// {
//     for(int i{0}; i != 100; ++i)
//     {
//         double ang{getRandomDouble(-1e-6, 1e-6)};
//         Eigen::Vector2d t{getRandomVector(-10.0, 10.0)};

//         SE2<double> T{SE2<double>::fromAngleAndVec(ang, t)};

//         Eigen::Matrix3d log_T{T.log()};
//         Eigen::Matrix3d log_T_true{T.T().log()};

//         EXPECT_TRUE(log_T_true.isApprox(log_T));
//     }
// }

// TEST(MatrixLogarith, SE2Element_Returns3Vector)
// {
//     for(int i{0}; i != 100; ++i)
//     {
//         SE2<double> T{SE2<double>::random()};

//         Eigen::Vector3d log_T{T.Log()};
//         Eigen::Vector3d log_T2{SE2<double>::Log(T)};
//         Eigen::Vector3d log_T3{SE2<double>::Log(T.T())};

//         Eigen::Matrix3d temp{T.T().log()};
//         Eigen::Vector3d log_T_true{SE2<double>::vee(temp)};

//         EXPECT_TRUE(log_T_true.isApprox(log_T));
//         EXPECT_TRUE(log_T_true.isApprox(log_T2));
//         EXPECT_TRUE(log_T_true.isApprox(log_T3));
//     }
// }

// TEST(MatrixExponential, se2Element_ReturnSE2Element)
// {
//     for(int i{0}; i != 100; ++i)
//     {
//         double ang{getRandomDouble(-PI, PI)};
//         Eigen::Vector2d t{getRandomVector(-10.0, 10.0)};
//         Eigen::Vector3d w;
//         w << ang, t(0), t(1);

//         Eigen::Matrix3d log_T{SE2<double>::hat(w)};

//         SE2<double> T{SE2<double>::exp(log_T)};
//         Eigen::Matrix3d T_true{log_T.exp()};

//         EXPECT_TRUE(T_true.isApprox(T.T()));
//     }
// }

// TEST(TaylorMatrixExponential, se2Element_ReturnsSE2Element)
// {
//     for(int i{0}; i != 100; ++i)
//     {
//         double ang{getRandomDouble(-1e-6, 1e-6)};
//         Eigen::Vector2d t{getRandomVector(-10.0, 10.0)};
//         Eigen::Vector3d w;
//         w << ang, t(0), t(1);

//         Eigen::Matrix3d log_T{SE2<double>::hat(w)};

//         SE2<double> T{SE2<double>::exp(log_T)};
//         Eigen::Matrix3d T_true{log_T.exp()};

//         EXPECT_TRUE(T_true.isApprox(T.T()));

//     }
// }

// TEST(MatrixExponential, 3Vector_ReturnsSE2Element)
// {
//     for(int i{0}; i != 100; ++i)
//     {
//         double ang{getRandomDouble(-PI, PI)};
//         Eigen::Vector2d t{getRandomVector(-10.0, 10.0)};
//         Eigen::Vector3d w;
//         w << ang, t(0), t(1);

//         SE2<double> T{SE2<double>::Exp(w)};

//         Eigen::Matrix3d log_T{SE2<double>::hat(w)};
//         Eigen::Matrix3d T_true{log_T.exp()};

//         EXPECT_TRUE(T_true.isApprox(T.T()));
//     }
// }

// TEST(Adjoint, SE2ElementAnd3Vector_ReturnsCorrectComposition)
// {
//     for(int i{0}; i != 100; ++i)
//     {
//         SE2<double> T{SE2<double>::random()};

//         double ang{getRandomDouble(-PI, PI)};
//         Eigen::Vector2d t{getRandomVector(-10.0, 10.0)};
//         Eigen::Vector3d delta;
//         delta << ang, t(0), t(1);

//         SE2<double> res1{T * SE2<double>::Exp(delta)};
//         Eigen::Vector3d temp{T.Adj() * delta};
//         SE2<double> temp2{SE2<double>::Exp(temp)};
//         SE2<double> res2{temp2 * T};

//         if(!(res1 == res2))
//         {
//             int debug = 1;
//             SE2<double> blah{temp2 * T};
//         }

//         EXPECT_TRUE(res1 == res2);
//     }
// }

// TEST(BoxPlus, SE2And3Vector_ReturnsNewSE2Element)
// {
//     for(int i{0}; i != 100; ++i)
//     {
//         SE2<double> T{SE2<double>::random()};
//         Eigen::Vector2d t{getRandomVector(-10.0, 10.0)};
//         double ang{getRandomDouble(-PI, PI)};
//         Eigen::Vector3d d;
//         d << t, ang;

//         SE2<double> res{T.boxplus(d)};
//         SE2<double> res_true{T * SE2<double>::Exp(d)};

//         EXPECT_TRUE(res == res_true);
//     }
// }

// TEST(BoxMinus, 2SE2Elements_ReturnsDifference)
// {
//     for(int i{0}; i != 100; ++i)
//     {
//         SE2<double> T1{SE2<double>::random()}, T2{SE2<double>::random()};

//         Eigen::Vector3d d{T1.boxminus(T2)};
//         SE2<double> T{T2.boxplus(d)};

//         EXPECT_TRUE(T == T1);
//     }
// }

// TEST(Identity, AskedForIdentity_ReturnsIdentity)
// {
//     Eigen::Matrix3d I{Eigen::Matrix3d::Identity()};
//     SE2<double> T{SE2<double>::Identity()};

//     EXPECT_TRUE(I.isApprox(T.T()));
// }
