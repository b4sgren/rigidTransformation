#include <gtest/gtest.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
// #include <eigen3/unsupported/Eigen/MatrixFunctions>
#include <random>

#include "rigidTransformation/quaternion.h"
#include "rigidTransformation/so3.h"
#include "rigidTransformation/utils.h"

namespace rt = rigidTransform;
using Quatd = rt::Quaternion<double>;

void fixQuat(Eigen::Vector4d &q) {
    if (q(0) < 0.0) q *= -1;
}

double randomDouble(double min, double max) {
    static std::random_device rd;
    static std::mt19937 generator(rd());
    static std::uniform_real_distribution<double> dist(min, max);

    return dist(generator);
}

Eigen::Vector3d getRandomVector(double min, double max) {
    Eigen::Vector3d v;
    v << randomDouble(min, max), randomDouble(min, max), randomDouble(min, max);

    return v;
}

void print(const Quatd &q1, const Quatd &q2) {
    std::cout << "----------------\n"
              << q1 << "\n"
              << q2 << std::endl;
}

class Quat_Fixture : public ::testing::Test {
   public:
    Quat_Fixture() {
        for (int i{0}; i != 100; ++i) transforms_.push_back(Quatd::random());
    }

    std::vector<Quatd> transforms_;
};

TEST_F(Quat_Fixture, DefaultInitialization) {
    Quatd q{};
    Eigen::Vector4d q_true(1, 0, 0, 0);
    EXPECT_TRUE(q_true.isApprox(q.q()));
}

TEST_F(Quat_Fixture, InitializeFromPointer) {
    double data[]{1, 0, 0, 0};
    Quatd q{data};
    Eigen::Map<Eigen::Vector4d> q_true(data);
    EXPECT_TRUE(q_true.isApprox(q.q()));
}

TEST_F(Quat_Fixture, InitializeFromVector) {
    Eigen::Vector4d q_true;
    q_true << 1, 0, 0, 0;
    Quatd q{q_true};
    EXPECT_TRUE(q_true.isApprox(q.q()));
}

TEST_F(Quat_Fixture, InitializeFromRPY) {
    for (int i{0}; i != 100; ++i) {
        double r{randomDouble(-rt::PI, rt::PI)};
        double p{randomDouble(-rt::PI, rt::PI)};
        double y{randomDouble(-rt::PI, rt::PI)};
        Quatd q{r, p, y};
        rt::SO3<double> R{r, p, y};

        EXPECT_TRUE(R.R().isApprox(q.R()));
    }
}

TEST_F(Quat_Fixture, InitializeFromRotationMatrix) {
    for (int i{0}; i != 100; ++i) {
        rt::SO3<double> R{rt::SO3<double>::random()};
        Quatd q(Quatd::fromR(R.R()));
        EXPECT_TRUE(R.R().isApprox(q.R()));
    }
}

TEST_F(Quat_Fixture, InitializeFromAxisAngle) {
    for (int i{0}; i != 100; ++i) {
        double theta{randomDouble(-rt::PI, rt::PI)};
        Eigen::Vector3d vec{getRandomVector(-10, 10)};
        vec = vec / vec.norm() * theta;

        Quatd q(Quatd::fromAxisAngle(vec));
        rt::SO3<double> R(rt::SO3<double>::fromAxisAngle(vec));

        EXPECT_TRUE(R.R().isApprox(q.R()));
    }
}

TEST_F(Quat_Fixture, InitializeFromQuaternion) {
    Eigen::Vector4d vec;
    vec << sqrt(2), 0, sqrt(2), 0;
    Quatd q{vec};
    Quatd q2{q};
    EXPECT_TRUE(q2.q().isApprox(q.q()));
}

TEST_F(Quat_Fixture, AssignmentOperator) {
    Eigen::Vector4d vec;
    vec << sqrt(2), 0, sqrt(2), 0;
    Quatd q{vec};
    Quatd q2 = q;
    EXPECT_TRUE(q2.q().isApprox(q.q()));
}

TEST_F(Quat_Fixture, RandomInitialization) {
    for (auto q : transforms_) EXPECT_FLOAT_EQ(1.0, q.norm());
}

TEST_F(Quat_Fixture, IdentityInitialization) {
    Quatd q{Quatd::Identity()};
    Eigen::Vector4d I;
    I << 1, 0, 0, 0;
    EXPECT_TRUE(I.isApprox(q.q()));
}

TEST_F(Quat_Fixture, GroupMultiplication) {
    for (Quatd q : transforms_) {
        Quatd q2{Quatd::random()};

        Quatd q3{q * q2};
        Eigen::Vector4d q3_true{Eigen::Vector4d::Zero()};
        q3_true(0) = q.qw() * q2.qw() - q.qx() * q2.qx() - q.qy() * q2.qy() -
                     q.qz() * q2.qz();
        q3_true(1) = q.qw() * q2.qx() + q.qx() * q2.qw() + q.qy() * q2.qz() -
                     q.qz() * q2.qy();
        q3_true(2) = q.qw() * q2.qy() - q.qx() * q2.qz() + q.qy() * q2.qw() +
                     q.qz() * q2.qx();
        q3_true(3) = q.qw() * q2.qz() + q.qx() * q2.qy() - q.qy() * q2.qx() +
                     q.qz() * q2.qw();

        if (q3_true(0) < 0) q3_true *= -1;

        EXPECT_TRUE(q3_true.isApprox(q3.q()));
    }
}

TEST_F(Quat_Fixture, OrderOfMultiplication) {
    for (int i{0}; i != 100; ++i) {
        double ang1{randomDouble(-rt::PI, rt::PI)},
            ang2{randomDouble(-rt::PI, rt::PI)};
        Eigen::Vector3d v{0, 0, 1};

        Eigen::Vector3d v1 = v * ang1;
        Quatd q_1_from_orig(Quatd::fromAxisAngle(v1));
        Eigen::Matrix3d R_1_from_origin(
            Eigen::AngleAxisd(ang1, Eigen::Vector3d::UnitZ()));

        v = Eigen::Vector3d(1, 0, 0);
        Eigen::Vector3d v2 = v * ang2;
        Quatd q_2_from_1(Quatd::fromAxisAngle(v2));
        Eigen::Matrix3d R_2_from_1(
            Eigen::AngleAxisd(ang2, Eigen::Vector3d::UnitX()));

        Eigen::Matrix3d resR(R_1_from_origin * R_2_from_1);
        Quatd resq(q_1_from_orig * q_2_from_1);

        Quatd resqR(Quatd::fromR(resR));
        EXPECT_TRUE(resq.q().isApprox(resqR.q()));
    }
}

TEST_F(Quat_Fixture, QuaternionInverse) {
    for (Quatd q : transforms_) {
        Quatd q_inv(q.inverse());
        Quatd res(q * q_inv);
        Quatd I(Quatd::Identity());

        EXPECT_TRUE(I.q().isApprox(res.q()));
    }
}

TEST_F(Quat_Fixture, InverseInPlace) {
    for (Quatd q : transforms_) {
        Quatd q_orig(q);
        q.inverse_();

        Quatd res(q * q_orig);
        Quatd I(Quatd::Identity());

        EXPECT_TRUE(I.q().isApprox(res.q()));
    }
}

TEST_F(Quat_Fixture, RotationOfAVector) {
    for (Quatd q : transforms_) {
        Eigen::Vector3d v(getRandomVector(-10, 10));
        Eigen::Vector3d vp(q.rotate<double, double>(v));
        Eigen::Vector3d vp_true(q.R() * v);

        EXPECT_TRUE(vp_true.isApprox(vp));
    }
}

TEST_F(Quat_Fixture, InverseRotationOfAVector) {
    for (Quatd q : transforms_) {
        Eigen::Vector3d v(getRandomVector(-10, 10));
        Eigen::Vector3d vp(q.inv_rotate<double, double>(v));
        Eigen::Vector3d vp_true(q.R().transpose() * v);

        EXPECT_TRUE(vp_true.isApprox(vp));
    }
}

TEST_F(Quat_Fixture, QuaternionExponentialMap) {
    for (int i{0}; i != 100; ++i) {
        Eigen::Vector3d v{getRandomVector(-rt::PI, rt::PI)};

        Quatd q(Quatd::Exp(v));
        rt::SO3<double> R(rt::SO3<double>::Exp(v));
        Quatd q_true(Quatd::fromR(R.R()));

        EXPECT_TRUE(q_true.q().isApprox(q.q()));
    }
}

TEST_F(Quat_Fixture, QuaternionLogarithmicMap) {
    for (Quatd q : transforms_) {
        Eigen::Vector3d logq{q.Log()};
        Quatd q2{Quatd::Exp(logq)};

        EXPECT_TRUE(q.q().isApprox(q2.q()));
    }
}

TEST_F(Quat_Fixture, BoxPlusr) {
    for (Quatd q : transforms_) {
        Eigen::Vector3d v{getRandomVector(-rt::PI, rt::PI)};
        Quatd q2(q.boxplusr<double, double>(v));
        Quatd q2_true(q * Quatd::Exp(v));

        EXPECT_TRUE(q2_true.q().isApprox(q2.q()));
    }
}

TEST_F(Quat_Fixture, Boxminusr) {
    for (Quatd q : transforms_) {
        Quatd q2(Quatd::random());
        Eigen::Vector3d v(q.boxminusr(q2));
        Quatd q_res(q2.boxplusr<double, double>(v));

        EXPECT_TRUE(q_res.q().isApprox(q.q()));
    }
}

TEST_F(Quat_Fixture, Boxplusl) {
    for (Quatd q : transforms_) {
        Eigen::Vector3d v(getRandomVector(-rt::PI, rt::PI));
        Quatd q2(q.boxplusl<double, double>(v));
        Quatd q2_true(Quatd::Exp(v) * q);

        EXPECT_TRUE(q2_true.q().isApprox(q2.q()));
    }
}

TEST_F(Quat_Fixture, Boxminusl) {
    for (Quatd q : transforms_) {
        Quatd q2(Quatd::random());
        Eigen::Vector3d v(q.boxminusl(q2));
        Quatd q_res(q2.boxplusl<double, double>(v));

        EXPECT_TRUE(q_res.q().isApprox(q.q()));
    }
}

TEST_F(Quat_Fixture, Normalize) {
    Eigen::Vector4d q{1, 0.1, 0, 0.1};
    Quatd quat(q);
    quat.normalize_();

    EXPECT_FLOAT_EQ(1, quat.norm());
}

TEST_F(Quat_Fixture, Adjoint) {
    for (Quatd q : transforms_) {
        Eigen::Vector3d v{getRandomVector(-rt::PI, rt::PI)};
        Quatd q2 = q.boxplusr<double, double>(v);
        Quatd q3 = q.boxplusl<double, double>(q.Adj() * v);

        EXPECT_TRUE(q2.q().isApprox(q3.q()));
    }
}

TEST_F(Quat_Fixture, Euler) {
    for (Quatd q : transforms_) {
        Eigen::Vector3d rpy = q.euler();
        Quatd q2 = Quatd(rpy(0), rpy(1), rpy(2));

        EXPECT_TRUE(q.q().isApprox(q2.q()));
    }
}
