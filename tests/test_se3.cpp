#include <gtest/gtest.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <eigen3/unsupported/Eigen/MatrixFunctions>
#include <random>
#include <vector>

#include "rigidTransformation/quaternion.h"
#include "rigidTransformation/se3.h"
#include "rigidTransformation/so3.h"
#include "rigidTransformation/utils.h"

namespace rt = rigidTransform;
using SE3d = rt::SE3<double>;
using Quatd = rt::Quaternion<double>;
using Vector7d = Eigen::Matrix<double, 7, 1>;

double randomDouble(double min, double max) {
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<double> dist(min, max);

    return dist(generator);
}

Eigen::Vector3d getRandomVector(double min, double max) {
    Eigen::Vector3d v;
    v << randomDouble(min, max), randomDouble(min, max), randomDouble(min, max);
    return v;
}

template <int n>
bool compareMat(const Eigen::Ref<const Eigen::Matrix<double, n, 1>> &v1,
                const Eigen::Ref<const Eigen::Matrix<double, n, 1>> &v2) {
    return v1.isApprox(v2);
}

class SE3_Fixture : public ::testing::Test {
   public:
    SE3_Fixture() {
        for (int i{0}; i != 100; ++i) transforms_.push_back(SE3d::random());
    }

    ~SE3_Fixture() {}

   protected:
    std::vector<SE3d> transforms_;
};

TEST_F(SE3_Fixture, DefaultInitialization) {
    SE3d T;
    Vector7d I(Vector7d::Zero());
    I(3) = 1.0;
    EXPECT_TRUE(compareMat<7>(I, T.T()));
    EXPECT_TRUE(compareMat<4>(I.tail<4>(), T.q()));
}

TEST_F(SE3_Fixture, PointerInitialization) {
    double data[]{1, 2, 3, 4, 5, 6, 7};
    SE3d T(data);
    Eigen::Map<Vector7d> T_true(data);
    EXPECT_TRUE(compareMat<7>(T_true, T.T()));
    EXPECT_TRUE(compareMat<4>(T_true.tail<4>(), T.q()));
}

TEST_F(SE3_Fixture, InitializeFromVector) {
    Vector7d T_true;
    T_true << 1, 2, 3, 4, 5, 6, 7;

    SE3d T(T_true);
    EXPECT_TRUE(compareMat<7>(T_true, T.T()));
    EXPECT_TRUE(compareMat<4>(T_true.tail<4>(), T.q()));
}

TEST_F(SE3_Fixture, InitializeFromRPY) {
    for (int i{0}; i != 100; ++i) {
        double r{randomDouble(-rt::PI, rt::PI)};
        double p{randomDouble(-rt::PI, rt::PI)};
        double y{randomDouble(-rt::PI, rt::PI)};
        Eigen::Vector3d t(getRandomVector(-10, 10));
        SE3d T(r, p, y, t);

        Quatd q{r, p, y};

        EXPECT_TRUE(compareMat<3>(t, T.t()));
        EXPECT_TRUE(compareMat<4>(q.q(), T.q()));
    }
}

TEST_F(SE3_Fixture, FromRotationMatrixAndt) {
    for (int i{0}; i != 100; ++i) {
        Quatd q(Quatd::random());
        Eigen::Vector3d t(getRandomVector(-10, 10));
        SE3d T(q.R(), t);

        EXPECT_TRUE(compareMat<3>(t, T.t()));
        EXPECT_TRUE(compareMat<4>(q.q(), T.q()));
    }
}

TEST_F(SE3_Fixture, FromQuatAndt) {
    for (int i{0}; i != 100; ++i) {
        Quatd q(Quatd::random());
        Eigen::Vector3d t(getRandomVector(-10, 10));

        SE3d T(q, t);
        EXPECT_TRUE(compareMat<3>(t, T.t()));
        EXPECT_TRUE(compareMat<4>(q.q(), T.q()));
    }
}

TEST_F(SE3_Fixture, FromAxisAngleAndt) {
    for (int i{0}; i != 100; ++i) {
        double theta{randomDouble(-rt::PI, rt::PI)};
        Eigen::Vector3d v(getRandomVector(-10, 10));
        v = v / v.norm() * theta;
        Eigen::Vector3d t(getRandomVector(-10, 10));

        Quatd q(q.fromAxisAngle(v));
        SE3d T(T.fromAxisAngleAndt(v, t));

        EXPECT_TRUE(compareMat<3>(t, T.t()));
        EXPECT_TRUE(compareMat<4>(q.q(), T.q()));
    }
}

TEST_F(SE3_Fixture, FromSE3) {
    Vector7d T_true;
    T_true << 1, 2, 3, 4, 5, 6, 7;
    SE3d T(T_true);
    SE3d T2(T);

    EXPECT_TRUE(compareMat<7>(T.T(), T2.T()));
}

TEST_F(SE3_Fixture, AssignmentOperator) {
    Vector7d T_true;
    T_true << 1, 2, 3, 4, 5, 6, 7;
    SE3d T(T_true);
    SE3d T2 = T;

    EXPECT_TRUE(compareMat<7>(T.T(), T2.T()));
}

TEST_F(SE3_Fixture, RandomInitialization) {
    for (int i{0}; i != 100; ++i) {
        SE3d T(SE3d::random());
        EXPECT_FLOAT_EQ(1.0, T.q().norm());
    }
}

TEST_F(SE3_Fixture, GroupMultiplication) {
    for (SE3d T : transforms_) {
        SE3d T2(SE3d::random());
        SE3d T3(T * T2);

        Eigen::Matrix4d TR, T2R, T3R;
        TR << T.R(), T.t(), 0, 0, 0, 1;
        T2R << T2.R(), T2.t(), 0, 0, 0, 1;

        T3R = TR * T2R;
        SE3d T3_true(T3R.block<3, 3>(0, 0), T3R.block<3, 1>(0, 3));

        EXPECT_TRUE(compareMat<3>(T3_true.t(), T3.t()));
        EXPECT_TRUE(compareMat<4>(T3_true.q(), T3.q()));
    }
}

TEST_F(SE3_Fixture, OrderOfGroupMultiplication) {
    for (int i{0}; i != 100; ++i) {
        Eigen::Vector3d t1(getRandomVector(-10, 10));
        Eigen::Vector3d t2(getRandomVector(-10, 10));
        Eigen::Vector3d vec(getRandomVector(-10, 10));
        vec = vec / vec.norm();
        double ang1(randomDouble(-rt::PI, rt::PI)),
            ang2(randomDouble(-rt::PI, rt::PI));

        SE3d T_1_from_orig(SE3d::fromAxisAngleAndt(vec * ang1, t1));
        SE3d T_2_from_1(SE3d::fromAxisAngleAndt(vec * ang2, t2));
        SE3d T_2_from_orig(T_2_from_1 * T_1_from_orig);

        Eigen::Matrix3d R_1_from_origin(
            rt::SO3<double>::fromAxisAngle(vec * ang1).R());
        Eigen::Matrix3d R_2_from_1(
            rt::SO3<double>::fromAxisAngle(vec * ang2).R());
        Eigen::Matrix4d TR_1_from_orig, TR_2_from_1;
        TR_1_from_orig << R_1_from_origin, t1, 0, 0, 0, 1;
        TR_2_from_1 << R_2_from_1, t2, 0, 0, 0, 1;
        Eigen::Matrix4d TR_2_from_orig = TR_2_from_1 * TR_1_from_orig;

        SE3d res(TR_2_from_orig.block<3, 3>(0, 0),
                 TR_2_from_orig.block<3, 1>(0, 3));
        EXPECT_TRUE(compareMat<3>(res.t(), T_2_from_orig.t()));
        EXPECT_TRUE(compareMat<4>(res.q(), T_2_from_orig.q()));
    }
}

TEST_F(SE3_Fixture, Inverse) {
    for (SE3d T : transforms_) {
        SE3d T_inv(T.inverse());
        SE3d T3(T * T_inv);
        Vector7d I(Vector7d::Zero());
        I(3) = 1.0;

        EXPECT_TRUE(compareMat<7>(I, T3.T()));
    }
}

TEST_F(SE3_Fixture, InverseInPlace) {
    for (SE3d T : transforms_) {
        SE3d T_orig(T);
        T.inverse_();
        SE3d res(T * T_orig);
        Vector7d I(Vector7d::Zero());
        I(3) = 1.0;

        EXPECT_TRUE(compareMat<7>(I, res.T()));
    }
}

TEST_F(SE3_Fixture, Transformation) {
    for (SE3d T : transforms_) {
        Eigen::Vector3d v(getRandomVector(-10, 10));
        Eigen::Vector3d vp(T.transform<double, double>(v));
        Eigen::Vector3d vp_true(T.t() + T.quat().rotate<double, double>(v));

        EXPECT_TRUE(compareMat<3>(vp_true, vp));
    }
}

TEST_F(SE3_Fixture, InverseTransformation) {
    for (SE3d T : transforms_) {
        Eigen::Vector3d v(getRandomVector(-10, 10));
        Eigen::Vector3d vp(T.inv_transform<double, double>(v));
        Eigen::Vector3d res(T.transform<double, double>(vp));

        EXPECT_TRUE(compareMat<3>(v, res));
    }
}

TEST_F(SE3_Fixture, ExponentialMap) {
    for (int i{0}; i != 100; ++i) {
        Eigen::Vector3d rho(getRandomVector(-10, 10));
        Eigen::Vector3d theta(getRandomVector(-rt::PI, rt::PI));
        Eigen::Matrix<double, 6, 1> tau;
        tau << rho, theta;

        SE3d T = SE3d::Exp(tau);
        Eigen::Matrix3d thetax = rt::skew3<double>(theta);
        Eigen::Matrix4d logT;
        logT << thetax, rho, 0, 0, 0, 0;
        Eigen::Matrix4d TR_true = logT.exp();

        SE3d T_true(TR_true.block<3, 3>(0, 0), TR_true.block<3, 1>(0, 3));

        EXPECT_TRUE(compareMat<3>(T_true.t(), T.t()));
        EXPECT_TRUE(compareMat<4>(T_true.q(), T.q()));
    }
}

TEST_F(SE3_Fixture, LogarithmicMap) {
    for (auto T : transforms_) {
        Eigen::Matrix<double, 6, 1> logT = T.Log();
        SE3d res = SE3d::Exp(logT);

        EXPECT_TRUE(compareMat<7>(T.T(), res.T()));
    }
}

TEST_F(SE3_Fixture, Boxplusr) {
    for (SE3d T : transforms_) {
        Eigen::Vector3d rho(getRandomVector(-10, 10));
        Eigen::Vector3d theta(getRandomVector(-rt::PI, rt::PI));
        Eigen::Matrix<double, 6, 1> tau;
        tau << rho, theta;

        SE3d T2(T.boxplusr<double, double>(tau));
        SE3d T_true = T * SE3d::Exp(tau);

        EXPECT_TRUE(compareMat<7>(T_true.T(), T2.T()));
    }
}

TEST_F(SE3_Fixture, Boxminusr) {
    for (SE3d T : transforms_) {
        SE3d T2(SE3d::random());
        Eigen::Matrix<double, 6, 1> diff(T.boxminusr<double, double>(T2));
        SE3d res(T2.boxplusr<double, double>(diff));

        EXPECT_TRUE(compareMat<7>(T.T(), res.T()));
    }
}

TEST_F(SE3_Fixture, Boxplusl) {
    for (SE3d T : transforms_) {
        Eigen::Vector3d rho(getRandomVector(-10, 10));
        Eigen::Vector3d theta(getRandomVector(-rt::PI, rt::PI));
        Eigen::Matrix<double, 6, 1> tau;
        tau << rho, theta;

        SE3d T2(T.boxplusl<double, double>(tau));
        SE3d T_true = SE3d::Exp(tau) * T;

        EXPECT_TRUE(compareMat<7>(T_true.T(), T2.T()));
    }
}

TEST_F(SE3_Fixture, Boxminusl) {
    for (SE3d T : transforms_) {
        SE3d T2(SE3d::random());
        Eigen::Matrix<double, 6, 1> diff(T.boxminusl<double, double>(T2));
        SE3d res(T2.boxplusl<double, double>(diff));

        EXPECT_TRUE(compareMat<7>(T.T(), res.T()));
    }
}

TEST_F(SE3_Fixture, Adjoint) {
    for (SE3d T : transforms_) {
        Eigen::Vector3d rho(getRandomVector(-10, 10));
        Eigen::Vector3d theta(getRandomVector(-rt::PI, rt::PI));
        Eigen::Matrix<double, 6, 1> tau;
        tau << rho, theta;

        SE3d T1(T.boxplusr<double, double>(tau));
        SE3d T2(T.boxplusl<double, double>(T.Adj() * tau));

        EXPECT_TRUE(compareMat<7>(T1.T(), T2.T()));
    }
}

TEST_F(SE3_Fixture, MatrixForm) {
    for (SE3d T : transforms_) {
        Eigen::Vector3d v(getRandomVector(-10, 10));
        Eigen::Vector4d vh;
        vh << v, 1;

        Eigen::Vector3d vp = T.transform<double, double>(v);
        Eigen::Vector4d vph = T.matrix() * vh;
        Eigen::Vector3d vp2 = vph.head<3>();

        EXPECT_TRUE(compareMat<3>(vp, vp2));
    }
}
