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

// TEST(SE2_Element, AskedForMatrix_ReturnsHomogeneousTransformationMatrix)
// {
//     Eigen::Matrix3d T;
//     T << 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0;
//     Eigen::Matrix2d R{T.block<2,2>(0,0)};
//     Eigen::Vector2d t{T.block<2,1>(0,2)};

//     SE2<double> T1{T};
//     EXPECT_TRUE(T.isApprox(T1.T()));
//     EXPECT_TRUE(R.isApprox(T1.R()));
//     EXPECT_TRUE(t.isApprox(T1.t()));
// }

// TEST(GenerateRandomSE2Element, IfTransformationIsValid_ReturnsTrue)
// {
//     for(int i{0}; i!=100; ++i)
//     {
//         SE2<double> T{SE2<double>::random()};
//         EXPECT_TRUE(T.isValidTransformation());
//     }
// }

// TEST(FromAngleAndVec, AngleAndTranslationVector_ReturnTransformationMatrix)
// {
//     for(int i{0}; i!=100; ++i)
//     {
//         double angle{getRandomDouble(-PI, PI)};
//         Eigen::Vector2d t{getRandomVector(-10.0, 10.0)};

//         SE2<double> T{SE2<double>::fromAngleAndVec(angle, t)};
//         double ct{cos(angle)}, st{sin(angle)};
//         Eigen::Matrix3d T_true;
//         T_true << ct, -st, t(0), st, ct, t(1), 0.0, 0.0, 1.0;

//         EXPECT_TRUE(T_true.isApprox(T.T()));
//     }
// }

// TEST(FromPointer, PointerInColumnMajorOrder_ReturnsTransformationMatrix)
// {
//     for(int i{0}; i != 100; ++i)
//     {
//         SE2<double> T{SE2<double>::random()};
//         SE2<double> T2{T.data()};

//         EXPECT_TRUE(T == T2);
//     }
// }

// TEST(GroupMultiplication, TwoSE2Objects_ReturnConcatenatedSE2Object)
// {
//     for(int i{0}; i != 100; ++i)
//     {
//         SE2<double> T1{SE2<double>::random()}, T2{SE2<double>::random()};

//         SE2<double> T3{T1 * T2};
//         Eigen::Matrix3d T3_true;
//         T3_true.block<2,2>(0,0) = T1.R() * T2.R();
//         T3_true.block<2,1>(0,2) = T1.t() + T1.R() * T2.t();
//         T3_true(2,2) = 1.0;

//         EXPECT_TRUE(T3_true.isApprox(T3.T()));
//     }
// }

// TEST(Inverse, SE2Object_ReturnInverse)
// {
//     for(int i{0}; i != 100; ++i)
//     {
//         SE2<double> T{SE2<double>::random()};
//         SE2<double> T_inv{T.inv()};

//         SE2<double> res{T * T_inv};

//         EXPECT_TRUE(Eigen::Matrix3d::Identity().isApprox(res.T()));
//     }
// }

// TEST(InverseInPlace, SE2Object_InvertsSelf)
// {
//     for(int i{0}; i != 100; ++i)
//     {
//         SE2<double> T{SE2<double>::random()};
//         Eigen::Matrix3d T_mat{T.T()};

//         T.selfInv();
//         Eigen::Matrix3d res{T.T() * T_mat};

//         EXPECT_TRUE(Eigen::Matrix3d::Identity().isApprox(res));
//     }
// }

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
