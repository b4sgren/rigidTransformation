#include <gtest/gtest.h>

#include <eigen3/Eigen/Core>
#include <iostream>

#include "rigidTransformation/se3.h"
#include "rigidTransformation/uncertain_compositions.h"
#include "rigidTransformation/utils.h"

namespace rt = rigidTransform;
using SE3d = rt::SE3<double>;
using Matrix6d = Eigen::Matrix<double, 6, 6>;

TEST(RightPerturbation, Inverse) {
    SE3d T = SE3d::random();
    Eigen::MatrixXd cov = Matrix6d::Identity();

    SE3d Tinv;
    Eigen::MatrixXd cov_inv = Matrix6d::Identity();
    rt::RightInvariant<SE3d>::inv(T, cov, Tinv, cov_inv);

    EXPECT_TRUE(cov_inv.isApprox(T.Adj() * T.Adj().transpose()));
}

TEST(RightPerturbation, Compose) {
    SE3d T1 = SE3d::random();
    SE3d T2 = SE3d::random();
    Eigen::MatrixXd cov1 = Matrix6d::Identity();
    Eigen::MatrixXd cov2 = Matrix6d::Identity();

    SE3d T3;
    Eigen::MatrixXd cov3;
    rt::RightInvariant<SE3d>::compose(T1, cov1, T2, cov2, T3, cov3);

    Eigen::MatrixXd res =
        T2.inverse().Adj() * T2.inverse().Adj().transpose() + cov2;
    EXPECT_TRUE(cov3.isApprox(res));
}

TEST(RightPerturbation, Between) {
    SE3d T1 = SE3d::random();
    SE3d T2 = SE3d::random();
    Eigen::MatrixXd cov1 = Matrix6d::Identity();
    Eigen::MatrixXd cov2 = Matrix6d::Identity();

    SE3d T3;
    Eigen::MatrixXd cov3;
    rt::RightInvariant<SE3d>::between(T1, cov1, T2, cov2, T3, cov3);

    Matrix6d Adj = T2.inverse().Adj() * T1.Adj();
    Eigen::MatrixXd res = Adj * Adj.transpose() + cov2;
    EXPECT_TRUE(cov3.isApprox(res));
}
