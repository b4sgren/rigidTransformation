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

TEST(TransformAVector, SE3ElementAnd3Vector_ReturnActivelyTransformedVector)
{
    for(int i{0}; i != 100; ++i)
    {
        SE3<double> T{SE3<double>::random()};
        Eigen::Vector3d v{getRandomVector(-10.0, 10.0)};
        Eigen::Vector4d vec;
        vec << v, 1.0;

        Eigen::Vector3d res{T.transa(v)};
        Eigen::Vector3d res2{T * v};
        Eigen::Vector3d res_true((T.T() * vec).head<3>());

        EXPECT_TRUE(res_true.isApprox(res));
        EXPECT_TRUE(res_true.isApprox(res2));
    }
}

TEST(TransformAVector, SE3ElementAndHomogeneousVector_ReturnActivelyTransformedHomogeneousVector)
{
    for(int i{0}; i != 100; ++i)
    {
        SE3<double> T{SE3<double>::random()};
        Eigen::Vector3d v{getRandomVector(-10.0, 10.0)};

        Eigen::Vector4d vec;
        vec << v, 1.0;

        Eigen::Vector4d res{T.transa(vec)};
        Eigen::Vector4d res2{T*vec};
        Eigen::Vector4d res_true{T.T() * vec};

        EXPECT_TRUE(res_true.isApprox(res));
        EXPECT_TRUE(res_true.isApprox(res2));
    }
}

TEST(TransformAVector, SE3ElementAnd3Vector_ReturnPasivelyTransformedVector)
{
    for(int i{0}; i != 100; ++i)
    {
        SE3<double> T{SE3<double>::random()};
        Eigen::Vector3d v{getRandomVector(-10.0, 10.0)};
        Eigen::Vector4d vec;
        vec << v, 1.0;

        Eigen::Vector3d res{T.transp(v)};
        Eigen::Vector3d res_true((T.T().inverse() * vec).head<3>());

        EXPECT_TRUE(res_true.isApprox(res));
    }
}

TEST(TransformAVector, SE3ElementAndHomogeneousVector_ReturnPassivelyTransformedHomogeneousVector)
{
    for(int i{0}; i != 100; ++i)
    {
        SE3<double> T{SE3<double>::random()};
        Eigen::Vector3d v{getRandomVector(-10.0, 10.0)};

        Eigen::Vector4d vec;
        vec << v, 1.0;

        Eigen::Vector4d res{T.transp(vec)};
        Eigen::Vector4d res_true{T.T().inverse() * vec};

        EXPECT_TRUE(res_true.isApprox(res));
    }
}

TEST(MatrixLogarithm, SE3Element_Returns4by4MatrixInLieAlgebra)
{
    for(int i{0}; i != 100; ++i)
    {
        SE3<double> T{SE3<double>::random()};

        Eigen::Matrix4d log_T_true{T.T().log()};
        Eigen::Matrix4d log_T{T.log()};
        Eigen::Matrix4d log_T2{SE3<double>::log(T)};
        Eigen::Matrix4d log_T3{SE3<double>::log(T.T())}; //Can call on not elements of the class

        double norm{(log_T_true - log_T).norm()};
        if(norm > 1e-8)
        {
            std::cout << "\n\n" << log_T_true << std::endl;
            std::cout << log_T << std::endl;
            std::cout << norm << std::endl;
        }

        EXPECT_TRUE(log_T2.isApprox(log_T));
        EXPECT_TRUE(log_T3.isApprox(log_T2));
        EXPECT_LE(norm, 1e-8);
    }
}

TEST(MatrixLogarithm, SE3Element_ReturnsMatInLieAlgebraUsingTaylorSeriesAbout0)
{
    for(int i{0}; i != 100; ++i)
    {
        double ang{getRandomDouble(0.0, 1e-6)};
        Eigen::Vector3d vec{getRandomVector(-10.0, 10.0)};
        vec  = vec / vec.norm() * ang;
        Eigen::Vector3d t{getRandomVector(-10.0, 10.0)};

        SE3<double> T{SE3<double>::fromAxisAngleAndVector(vec, t)};

        Eigen::Matrix4d log_T_true{T.T().log()};
        Eigen::Matrix4d log_T{T.log()};

        double norm{(log_T_true - log_T).norm()};

        if(norm > 1e-6 || std::isnan(norm))
        {
            std::cout << "\n\n" << log_T_true << std::endl;
            std::cout << log_T << std::endl;
            std::cout << norm << std::endl;
            Eigen::Matrix4d temp{T.log()};
        }

        EXPECT_LE(norm, 1e-6);
    }
}

TEST(MatrixLogarithm, DISABLED_SE3Element_ReturnsMatInLieAlgebraUsingTaylorSeriesAboutPI)
{
    for(int i{0}; i != 100; ++i)
    {
        double ang{PI - getRandomDouble(0.0, 1e-6)};
        Eigen::Vector3d vec{getRandomVector(-10.0, 10.0)};
        vec  = vec / vec.norm() * ang;
        Eigen::Vector3d t{getRandomVector(-10.0, 10.0)};

        SE3<double> T{SE3<double>::fromAxisAngleAndVector(vec, t)};

        Eigen::Matrix4d log_T_true{T.T().log()};
        Eigen::Matrix4d log_T{T.log()};

        double norm{(log_T_true - log_T).norm()};

        if(norm > 1e-8)
        {
            std::cout << "\n\n" << log_T_true << std::endl;
            std::cout << log_T << std::endl;
            std::cout << norm << std::endl;
        }

        EXPECT_LE(norm, 1e-8); //Haven't worked this out in python either
    }
}

TEST(MatrixExponential, ElementOfSE3LieAlgebra_ReturnSE3Element)
{
    for(int i{0}; i != 100; ++i)
    {
        Eigen::Vector3d w{getRandomVector(-PI, PI)}, t{getRandomVector(-10.0, 10.0)};
        Eigen::Matrix4d logT;
        logT << skew3(w), t, Eigen::RowVector4d::Zero();

        Eigen::Matrix4d T_true{logT.exp()};
        SE3<double> T{SE3<double>::exp(logT)};

        EXPECT_TRUE(T_true.isApprox(T.T()));
    }
}

TEST(MatrixExponential, ElementOfSE3LieAlgebra_ReturnsSE3ElementUsingTaylorSeriesAobut0)
{
    for(int i{0}; i != 100; ++i)
    {
        double ang{getRandomDouble(0.0, 1e-6)};
        Eigen::Vector3d w{getRandomVector(-10.0, 10.0)}, t{getRandomVector(-10.0, 10.0)};
        w = w / w.norm() * ang;

        Eigen::Matrix4d logT;
        logT << skew3(w), t, Eigen::RowVector4d::Zero();

        Eigen::Matrix4d T_true{logT.exp()};
        SE3<double> T{SE3<double>::exp(logT)};

        EXPECT_TRUE(T_true.isApprox(T.T()));
    }
}

TEST(HatOperator, SixVector_Returns4x4MatrixOfLieAlgebra)
{
    for(int i{0}; i != 100; ++i)
    {
        Eigen::Vector3d w{getRandomVector(-PI, PI)}, u{getRandomVector(-10.0, 10.0)};
        Eigen::Matrix<double, 6, 1> xci;
        xci << u, w;

        Eigen::Matrix4d logT{SE3<double>::hat(xci)};
        Eigen::Matrix4d log_T_true;
        log_T_true << 0, -w(2), w(1), u(0), w(2), 0, -w(0), u(1), -w(1), w(0), 0, u(2), 0, 0, 0, 0;

        EXPECT_TRUE(log_T_true.isApprox(logT));
    }
}

TEST(VeeOperator, 4x4Matrix_Returns6Vector)
{
    for(int i{0}; i != 100; ++i)
    {
        Eigen::Vector3d w{getRandomVector(-PI, PI)}, u{getRandomVector(-10.0, 10.0)};
        Eigen::Matrix<double,6,1> xci_true;
        xci_true << u, w;

        Eigen::Matrix4d logT{SE3<double>::hat(xci_true)};
        Eigen::Matrix<double,6,1> xci{SE3<double>::vee(logT)};

        EXPECT_TRUE(xci_true.isApprox(xci));
    }
}

TEST(MatrixLogarithm, ElementOfSE3_ReturnsA6Vector)
{
    for(int i{0}; i != 100; ++i)
    {
        SE3<double> T{SE3<double>::random()};

        Eigen::Matrix<double,6,1> xci{T.Log()}, xci_true; //Same 3 calls as other log method
        Eigen::Matrix4d logT{T.T().log()};

        xci_true << logT.block<3,1>(0,3), logT(2,1), logT(0,2), logT(1,0);

        double norm{(xci_true - xci).norm()};

        EXPECT_LT(norm, 1e-8);
    }
}

TEST(MatrixExponential, 6Vector_ReturnElementOfSE3)
{
    for(int i{0}; i != 100; ++i)
    {
        Eigen::Vector3d w{getRandomVector(-PI, PI)}, t{getRandomVector(-10.0, 10.0)};
        Eigen::Matrix<double,6,1> xci;
        xci << t, w;
        Eigen::Matrix4d logT;
        logT << 0, -w(2), w(1), t(0), w(2), 0, -w(0), t(1), -w(1), w(0), 0, t(2), 0, 0, 0, 0;

        Eigen::Matrix4d T_true{logT.exp()};
        SE3<double> T{SE3<double>::Exp(xci)};

        EXPECT_TRUE(T_true.isApprox(T.T()));
    }
}

TEST(BoxPlus, SE3ElementAnd6Vector_ReturnsSE3ElementWhenAdded)
{
    for(int i{0}; i != 100; ++i)
    {
        SE3<double> T{SE3<double>::random()};
        Eigen::Vector3d w{getRandomVector(-PI, PI)}, t{getRandomVector(-10.0, 10.0)};
        Eigen::Matrix<double,6,1> xci;
        xci << t, w;

        SE3<double> T_res{T.boxplus(xci)};
        Eigen::Matrix4d T_true{T.T() * SE3<double>::hat(xci).exp()};
    }
}

TEST(BoxMinus, ElementsOfSE3_6VectorRepresentingTheDifference)
{
    for(int i{0}; i != 100; ++i)
    {
        SE3<double> T1{SE3<double>::random()}, T2{SE3<double>::random()};

        Eigen::Matrix<double,6,1>xci{T1.boxminus(T2)};

        SE3<double> T{T2.boxplus(xci)};

        if(!(T == T1))
        {
            std::cout << "T1:\n" << T1.T() << std::endl;
            std::cout << "T2:\n" << T.T() << std::endl;
        }

        EXPECT_TRUE(T == T1);
    }
}

TEST(Adjoint, ElementOfSE3And6Vector_MultipliedOnLeftEqualsMultipliedOnRight)
{
    for(int i{0}; i != 100; ++i)
    {
        SE3<double> T1{SE3<double>::random()};
        Eigen::Vector3d w{getRandomVector(-PI, PI)}, t{getRandomVector(-10.0, 10.0)};
        Eigen::Matrix<double,6,1> xci;
        xci << t, w;

        SE3<double> T2{T1.boxplus(xci)};
        SE3<double> T3{SE3<double>::Exp(T1.Adj() * xci) * T1};

        EXPECT_TRUE(T2 == T3);
    }
}

TEST(Identity, AskedForIdentity_ReturnsIdentity)
{
    Eigen::Matrix4d I{Eigen::Matrix4d::Identity()};
    SE3<double> T{SE3<double>::Identity()};

    EXPECT_TRUE(I.isApprox(T.T()));
}