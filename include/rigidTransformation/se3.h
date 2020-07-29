#ifndef SE3_H
#define SE3_H

#include <random>
#include <iostream>
#include <cmath>

#include <eigen3/Eigen/Dense>

#include "utils.h"

template<typename F>
class SE3
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Mat4F = Eigen::Matrix<F,4,4>;
    using Mat3F = Eigen::Matrix<F,3,3>;
    using Vec3F = Eigen::Matrix<F,3,1>;
    using Vec6F = Eigen::Matrix<F,6,1>;
public:
    SE3(): _arr{Mat4F::Identity()} {}
    SE3(const Mat4F &mat): _arr{mat} {}

    Mat4F T() { return _arr; }
    Mat3F R() { return _arr.template block<3,3>(0,0); }
    Vec3F t() { return _arr.template block<3,1>(0,3); }

    bool isValidTransformation()
    {
        F det{this->R().determinant()};
        bool homogeneous(_arr(3,3) == F(1.0));
        // bool zeros(_arr.template .block<1,3>(3,0).isApprox(Vec3F::Zero().transpose()));
        bool one{abs(det - 1.0) < 1e-8};

        return one && homogeneous;
    }

    static SE3 random()
    {
        static std::random_device rd;
        static std::mt19937 generator(rd());
        static std::uniform_real_distribution<double> trans_dist(-10.0, 10.0);
        static std::uniform_real_distribution<double> dist(0, 1);

        Vec3F t;
        t << trans_dist(generator), trans_dist(generator), trans_dist(generator);

        Vec3F x;
        x << dist(generator), dist(generator), dist(generator);

        F psi{ 2 * PI * x(0)};
        Mat3F R;
        R << cos(psi), sin(psi), F(0), -sin(psi), cos(psi), F(0), F(0), F(0), F(1);

        Vec3F v;
        v << cos(2 * PI * x(1)) * sqrt(x(2)), sin(2 * PI * x(1)) * sqrt(x(2)), sqrt(F(1) - x(2));

        Mat3F H{Mat3F::Identity() - 2 * v * v.transpose()};
        R = -H * R;

        Mat4F T;
        T << R, t, Vec3F::Zero().transpose(), F(1);

        return SE3(T);
    }

private:
    Mat4F _arr;
};

#endif