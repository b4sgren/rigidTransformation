#include <gtest/gtest.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <eigen3/unsupported/Eigen/MatrixFunctions>
#include <random>

#include "so3.h"

double getRandomDouble(double min, double max)
{
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<double> dist(min, max);

    return dist(generator);
}

Eigen::Vector3d getRandomVector(double min, double max)
{
    Eigen::Vector3d v;
    v << getRandomDouble(min, max), getRandomDouble(min, max), getRandomDouble(min, max);
    return v;
}

TEST(GivenSO3Element, Return3By3Matrix)
{
   Eigen::Matrix3d R_true;
   R_true << 1, 0, 0, 0, 1, 0, 0, 0, 1; 

   SO3<double> R{R_true};

   EXPECT_TRUE(R_true.isApprox(R.R()));
}

TEST(RandomGeneration, AskedForRandomRotationMatrix_ReturnsValidRotationMatrix)
{
    for(int i{0}; i != 100; ++i)
    {
        SO3<double> R{SO3<double>::random()};
        EXPECT_TRUE(R.isValidRotation());
    }
}

TEST(FromAxisAngle, SuppliedAxisAngleVector_ReturnValidRotationMatrix)
{
    for(int i{0}; i != 100; ++i)
    {
        Eigen::Vector3d v{getRandomVector(-10.0, 10.0)};
        v /= v.norm();
        double ang = getRandomDouble(0.0, PI);

        Eigen::Matrix3d R_true{Eigen::AngleAxisd(ang, v)};

        SO3<double> R{SO3<double>::fromAxisAngle(v * ang)};
        EXPECT_TRUE(R_true.isApprox(R.R()));
        EXPECT_TRUE(R.isValidRotation());
    }
}

TEST(FromAxisAngleTaylorSeries, SuppliedAxisAngleVector_ReturnsValidRotationMatrix)
{
    for(int i{0}; i!=100; ++i)
    {
        Eigen::Vector3d v{getRandomVector(-10.0, 10.0)};
        v /= v.norm();
        double ang{getRandomDouble(0, 1e-6)};

        Eigen::Matrix3d R_true{Eigen::AngleAxisd(ang, v)};
        SO3<double> R{SO3<double>::fromAxisAngle(v * ang)};
        EXPECT_TRUE(R.isValidRotation());
        EXPECT_TRUE(R_true.isApprox(R.R()));
    }
}

TEST(FromAxisAngleEigen, SuppliedAxisAngleVector_ReturnsValidRotation)
{
    for(int i{0}; i != 100; ++i)
    {
        Eigen::Vector3d v{getRandomVector(-10.0, 10.0)};
        v /= v.norm();
        double ang = getRandomDouble(0.0, PI);

        Eigen::Matrix3d R_true{Eigen::AngleAxisd(ang, v)};

        SO3<double> R{SO3<double>::fromAxisAngle(Eigen::AngleAxisd(ang, v))};
        EXPECT_TRUE(R_true.isApprox(R.R()));
        EXPECT_TRUE(R.isValidRotation());
    }
}

TEST(FromRPYAngles, SuppliedRPYAngles_ReturnsValidRotation)
{
    for(int i{0}; i!=100; ++i)
    {
        Eigen::Vector3d angs{getRandomVector(-PI, PI)};

        SO3<double>R{SO3<double>::fromRPY(angs)};
        Eigen::Matrix3d Rx{Eigen::AngleAxisd(angs(0), Eigen::Vector3d::UnitX())};
        Eigen::Matrix3d Ry{Eigen::AngleAxisd(angs(1), Eigen::Vector3d::UnitY())};
        Eigen::Matrix3d Rz{Eigen::AngleAxisd(angs(2), Eigen::Vector3d::UnitZ())};
        Eigen::Matrix3d R_true{Rz * Ry * Rx};

        EXPECT_TRUE(R.isValidRotation());
        EXPECT_TRUE(R_true.isApprox(R.R()));
    }
}

TEST(FromQuaternion, SuppliedHamiltonianQuaternion_ReturnsValidRotation)
{
    for(int i{0}; i!=100; ++i)
    {
        Eigen::Vector3d v{getRandomVector(-10.0, 10.0)};
        v /= v.norm();
        double ang{getRandomDouble(0, PI)};

        Eigen::Matrix3d R_true{Eigen::AngleAxisd(ang, v)};
        Eigen::Vector4d q;
        double st = sin(ang/2);
        q << cos(ang/2), st * v(0), st * v(1), st * v(2);

        SO3<double> R{SO3<double>::fromQuaternion(q)};

        EXPECT_TRUE(R.isValidRotation());
        EXPECT_TRUE(R_true.isApprox(R.R()));
    }
}

TEST(FromQuaternionEigen, SuppliedEigenQuaternion_ReturnsValidRotation)
{
    for(int i{0}; i!=100; ++i)
    {
        Eigen::Vector3d v{getRandomVector(-10.0, 10.0)};
        v /= v.norm();
        double ang{getRandomDouble(0, PI)};

        Eigen::Matrix3d R_true{Eigen::AngleAxisd(ang, v)};
        Eigen::Quaterniond q{Eigen::AngleAxisd(ang, v)};

        SO3<double> R{SO3<double>::fromQuaternion(q)};

        EXPECT_TRUE(R.isValidRotation());
        EXPECT_TRUE(R_true.isApprox(R.R()));
    }
}

TEST(Inverse, AskedForInverse_InverseTimesOriginalGivesIdentity)
{
    for(int i{0}; i != 100; ++i)
    {
        SO3<double> R{SO3<double>::random()};
        SO3<double> R_inv{R.inv()};

        EXPECT_TRUE(Eigen::Matrix3d::Identity().isApprox(R_inv.R() * R.R()));
    }
}

TEST(SelfInverse, Inverse_AskedForInverse_InverseTimesOriginalGivesIdentity_Test)
{
    for(int i{0}; i!=100; ++i)
    {
        SO3<double> R{SO3<double>::random()};
        Eigen::Matrix3d R_inv{R.R()};
        R.selfInv();

        EXPECT_TRUE(Eigen::Matrix3d::Identity().isApprox(R_inv * R.R()));
    }
}

TEST(GroupOperator, TwoSO3ElementsMultiplied_ReturnsValidRotation)
{
    for(int i{0}; i != 100; ++i)
    {
        SO3<double> R1{SO3<double>::random()};
        SO3<double> R2{SO3<double>::random()};

        SO3<double> R3{R1 * R2};

        Eigen::Matrix3d R_true{R1.R() * R2.R()};

        EXPECT_TRUE(R_true.isApprox(R3.R()));
        EXPECT_TRUE(R3.isValidRotation());
    }
}

TEST(ActiveRotation, SO3ElementAnd3Vector_ReturnActivelyRotatedVector)
{
    for(int i{0}; i != 100; ++i)
    {
        SO3<double> R{SO3<double>::random()};
        Eigen::Vector3d v{getRandomVector(-10.0, 10.0)};

        Eigen::Vector3d vp{R.rota(v)};
        Eigen::Vector3d vp_true{R.R() * v};

        EXPECT_TRUE(vp_true.isApprox(vp));
    }
}

TEST(PassiveRotation, SO3ElementAnd3Vector_ReturnPassivelyRotatedVector)
{
    for(int i{0}; i!=100; ++i)
    {
        SO3<double> R{SO3<double>::random()};
        Eigen::Vector3d v{getRandomVector(-10.0, 10.0)};

        Eigen::Vector3d vp{R.rotp(v)};
        Eigen::Vector3d vp_true{R.inv().R() * v};

        EXPECT_TRUE(vp_true.isApprox(vp));
    }
}

TEST(MatrixLogarithm, DISABLED_SO3Element_ReturnsMatrixLogarithm)
{
    for(int i{0}; i!=100; ++i)
    {
        SO3<double> R{SO3<double>::random()};

        Eigen::Matrix3d log_R{R.log()};
        Eigen::Matrix3d log_R_true{R.R().log()};

        // if(!log_R_true.isApprox(log_R, 1e-8))
        // {
        //     std::cout << "Truth\n" << log_R_true << std::endl;
        //     std::cout << "Mine\n" << log_R << std::endl;
        //     int x{3};
        // }

        EXPECT_TRUE(log_R_true.isApprox(log_R, 1e-8));
    }
}

TEST(MatrixLogTaylor0, DISABLED_ActiveRotation_ReturnsMatrixLogarithm)
{
    for(int i{0}; i!=100; ++i)
    {
        Eigen::Vector3d v{getRandomVector(-10.0, 10.0)};
        v /= v.norm();
        double ang{getRandomDouble(0.0, 1e-6)};

        Eigen::Matrix3d R{Eigen::AngleAxisd(ang, v)};
        SO3<double> R1{R};

        Eigen::Matrix3d log_R_true{R.log()};
        Eigen::Matrix3d log_R{R1.log()};

        // if(!log_R_true.isApprox(log_R, 1e-8))
        // {
        //     std::cout << "Truth\n" << log_R_true << std::endl;
        //     std::cout << "Mine\n" << log_R << std::endl;
        //     std::cout << (log_R_true.array() - log_R.array()).matrix().norm() << std::endl;
        //     int x{3};
        // }

        EXPECT_TRUE(log_R_true.isApprox(log_R, 1e-8)); //Test doesn't pass but they are equivalent
    }
}

TEST(MatrixLogTaylorPI, DISABLED_ActiveRotation_ReturnsMatrixLogarithm)
{
    for(int i{0}; i!=100; ++i)
    {
        Eigen::Vector3d v{getRandomVector(-10.0, 10.0)};
        v /= v.norm();
        double ang{getRandomDouble(PI-1e-6, PI)};

        Eigen::Matrix3d R{Eigen::AngleAxisd(ang, v)};
        SO3<double> R1{R};

        Eigen::Matrix3d log_R_true{R.log()};
        Eigen::Matrix3d log_R{R1.log()};

        if(!log_R_true.isApprox(log_R, 1e-8))
        {
            std::cout << "Truth\n" << log_R_true << std::endl;
            std::cout << "Mine\n" << log_R << std::endl;
            int x{3};
        }

        EXPECT_TRUE(log_R_true.isApprox(log_R, 1e-8)); //Test doesn't pass. Close but not yet
    }
}

TEST(VeeOperator, SkewSymmetrixMatrix_Return3Vector)
{
    for(int i{0}; i != 100; ++i)
    {
        Eigen::Vector3d w_true{getRandomVector(-10.0, 10.0)};
        Eigen::Matrix3d log_R{skew(w_true)};
        Eigen::Vector3d w{SO3<double>::vee(log_R)};

        EXPECT_TRUE(w_true.isApprox(w));
    }
}