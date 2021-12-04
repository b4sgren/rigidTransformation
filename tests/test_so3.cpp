#include <gtest/gtest.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <eigen3/unsupported/Eigen/MatrixFunctions>
#include <random>

#include "so3.h"
#include "quaternion.h"
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
    rt::SO3<double> R(vals);
    Eigen::Matrix3d M;
    M << 1, 2, 3, 4, 5, 6, 7, 8, 9;
    M.transposeInPlace();

    EXPECT_TRUE(M.isApprox(R.R()));
}

TEST_F(SO3_Fixture, TestInitializationFromMatrix)
{
    Eigen::Matrix3d M{Eigen::Matrix3d::Random()};
    rt::SO3<double> R(M);

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

        rt::SO3<double> R(rt::SO3<double>::fromAxisAngle(v));
        Eigen::Matrix3d R_true = Eigen::AngleAxisd(phi, vec/vec.norm()).matrix();

        EXPECT_TRUE(R_true.isApprox(R.R()));
    }
}

TEST_F(SO3_Fixture, TestFromQuaternion) {
    for (int i{0}; i != 100; ++i) {
        rt::Quaternion<double> q = rt::Quaternion<double>::random();
        rt::SO3<double> R = rt::SO3<double>::fromQuat(q.q());

        EXPECT_TRUE(R.R().isApprox(q.R().transpose()));
    }
}

TEST_F(SO3_Fixture, InitializeFromSO3)
{
    Eigen::Matrix3d M{Eigen::Matrix3d::Random()};
    rt::SO3<double> R(M);
    rt::SO3<double> R2(R);

    EXPECT_TRUE(R.R().isApprox(R2.R()));
}

TEST_F(SO3_Fixture, TestAssignmentOperator)
{
    Eigen::Matrix3d M{Eigen::Matrix3d::Random()};
    rt::SO3<double> R(M);
    rt::SO3<double> R2 = R;

    EXPECT_TRUE(R.R().isApprox(R2.R()));
}

TEST_F(SO3_Fixture, TestRandomInitialization)
{
    for(int i{0}; i != 100; ++i)
    {
        rt::SO3<double> R(rt::SO3<double>::random());
        EXPECT_FLOAT_EQ(1.0, R.R().determinant());
    }
}

TEST_F(SO3_Fixture, TestIdentityFunction)
{
    rt::SO3<double> R(rt::SO3<double>::Identity());
    Eigen::Matrix3d I{Eigen::Matrix3d::Identity()};
    EXPECT_TRUE(I.isApprox(R.R()));
}

TEST_F(SO3_Fixture, TestGroupMultiplication)
{
    for(auto R : transforms_)
    {
        rt::SO3<double> R2(rt::SO3<double>::random());
        rt::SO3<double> R3(R * R2);

        Eigen::Matrix3d R3_true = R.R() * R2.R();

        EXPECT_TRUE(R3_true.isApprox(R3.R()));
    }
}

TEST_F(SO3_Fixture, TestInverse)
{
    Eigen::Matrix3d I{Eigen::Matrix3d::Identity()};
    for(auto R : transforms_)
    {
        rt::SO3<double> R_inv(R.inverse());
        rt::SO3<double> res(R * R_inv);
        EXPECT_TRUE(I.isApprox(res.R()));
    }
}

TEST_F(SO3_Fixture, TestInPlaceInverse)
{
    Eigen::Matrix3d I{Eigen::Matrix3d::Identity()};
    for(auto R : transforms_)
    {
        rt::SO3<double> R_orig(R);
        R.inverse_();
        rt::SO3<double> res(R * R_orig);
        EXPECT_TRUE(I.isApprox(res.R()));
    }
}

TEST_F(SO3_Fixture, TestActiveRotation)
{
    for(auto R : transforms_)
    {
        Eigen::Vector3d v{getRandomVector(-10.0, 10.0)};
        Eigen::Vector3d vp{R.rota<double>(v)};
        Eigen::Vector3d vp_true{R.R() * v};

        EXPECT_TRUE(vp_true.isApprox(vp));
    }
}

TEST_F(SO3_Fixture, TestPassiveRotation)
{
    for(auto R : transforms_)
    {
        Eigen::Vector3d v{getRandomVector(-10.0, 10.0)};
        Eigen::Vector3d vp{R.rotp<double>(v)};
        Eigen::Vector3d vpp{R.rota<double>(vp)};

        EXPECT_TRUE(v.isApprox(vpp));
    }
}

TEST_F(SO3_Fixture, TestLogarithmicMap)
{
    for(auto R : transforms_)
    {
        Eigen::Vector3d logR{R.Log()};
        Eigen::Matrix3d temp{R.R().log()};
        Eigen::Vector3d logR_true;
        logR_true << temp(2,1), temp(0,2), temp(1,0);

        EXPECT_TRUE(logR_true.isApprox(logR, 1e-8));
    }
}

TEST_F(SO3_Fixture, ExponentialMap)
{
    for(auto R : transforms_)
    {
        Eigen::Vector3d logR{R.Log()};
        rt::SO3<double> R2(rt::SO3<double>::Exp(logR));
        EXPECT_TRUE(R.R().isApprox(R2.R(), 1e-8));
    }
}

TEST_F(SO3_Fixture, TestAdjoint)
{
    for(auto R : transforms_)
    {
        Eigen::Vector3d v{getRandomVector(-10.0, 10.0)};

        rt::SO3<double> R2(R * rt::SO3<double>::Exp(v));
        rt::SO3<double> R3(rt::SO3<double>::Exp(R.Adj() * v) * R);
        EXPECT_TRUE(R2.R().isApprox(R3.R()));
    }
}

TEST_F(SO3_Fixture, Boxplusr)
{
    for(auto R : transforms_)
    {
        double theta{getRandomDouble(-rt::PI, rt::PI)};
        Eigen::Vector3d v{getRandomVector(-10, 10)};
        v = v/v.norm() * theta;
        rt::SO3<double> R2(R.boxplusr<double>(v));
        rt::SO3<double> R3(R * rt::SO3<double>::fromAxisAngle(v));

        EXPECT_TRUE(R2.R().isApprox(R3.R()));
    }
}

TEST_F(SO3_Fixture, Boxminusr)
{
    for(auto R : transforms_)
    {
        rt::SO3<double> R2{rt::SO3<double>::random()};
        Eigen::Vector3d diff{R.boxminusr(R2)};
        rt::SO3<double> R3{R2.boxplusr<double>(diff)};

        EXPECT_TRUE(R.R().isApprox(R3.R(), 1e-8));
    }
}

TEST_F(SO3_Fixture, Boxplusl)
{
    for(auto R : transforms_)
    {
        double theta{getRandomDouble(-rt::PI, rt::PI)};
        Eigen::Vector3d v{getRandomVector(-10, 10)};
        v = v/v.norm() * theta;
        rt::SO3<double> R2(R.boxplusl<double>(v));
        rt::SO3<double> R3(rt::SO3<double>::fromAxisAngle(v) * R);

        EXPECT_TRUE(R2.R().isApprox(R3.R()));
    }

}

TEST_F(SO3_Fixture, Boxminusl)
{
    for(auto R : transforms_)
    {
        rt::SO3<double> R2(rt::SO3<double>::random());
        Eigen::Vector3d diff{R.boxminusl(R2)};
        rt::SO3<double> R3(R2.boxplusl<double>(diff));

        EXPECT_TRUE(R.R().isApprox(R3.R(), 1e-8));
    }
}

TEST_F(SO3_Fixture, Euler) {
    for (auto R : transforms_) {
        Eigen::Vector3d rpy = R.euler();
        rt::SO3<double> R2 = rt::SO3<double>(rpy(0), rpy(1), rpy(2));

        EXPECT_TRUE(R.R().isApprox(R2.R()));
    }
}

// TEST(Normalize, GivenSO3ElementWithDetNotEq1_NormalizeElement)
// {
//     SO3<double> R1{SO3<double>::random()}, R2{SO3<double>::random()};

//     // while(R1.isValidRotation()) //Really slow to not be valid
//         // R1 *= R2;

//     R1.normalize();

//     EXPECT_TRUE(R1.isValidRotation());
// }
