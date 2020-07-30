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
    SE3(const Mat3F &R, const Vec3F &t)
    {
        _arr = Mat4F::Identity();
        _arr.template block<3,3>(0,0) = R;
        _arr.template block<3,1>(0,3) = t;
    }

    bool operator==(const SE3 &T)
    {
        return this->T().isApprox(T.T());
    }

    Mat4F T() const { return _arr; }
    Mat3F R() const { return _arr.template block<3,3>(0,0); }
    Vec3F t() const { return _arr.template block<3,1>(0,3); }

    bool isValidTransformation() const
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

        return SE3(R, t);
    }

    static SE3 fromAxisAngleAndVector(const Vec3F &vec, const Vec3F &t)
    {
        F theta{vec.norm()};

        F A, B;
        if(theta > 1e-6)
        {
            A = sin(theta)/theta;
            B = (1 - cos(theta))/pow(theta,2);
        }
        else 
        {
            A = 1 - pow(theta,2)/6.0 + pow(theta,4)/120.0;
            B = 0.5 - pow(theta,2)/24.0 + pow(theta,4)/720.0;
        }
        
        Mat3F skew_v{skew3(vec)};
        Mat3F R{Mat3F::Identity() + A * skew_v + B * skew_v * skew_v};

        return SE3(R, t);
    }

    static SE3 fromAxisAngleAndVector(const Eigen::AngleAxis<F> &v, const Vec3F &t)
    {
        Mat3F R(v);
        return SE3(R, t);
    }

    static SE3 fromRPYAndVector(const F &phi, const F &theta, const F &psi, const Vec3F &t)
    {
        F cp{cos(phi)}, sp{sin(phi)};
        F ct{cos(theta)}, st{sin(theta)};
        F cpsi{cos(psi)}, spsi{sin(psi)};

        Mat3F Rx, Ry, Rz;
        Rx << F(1.0), F(0.0), F(0.0), F(0.0), cp, -sp, F(0.0), sp, cp;
        Ry << ct, F(0.0), st, F(0.0), F(1.0), F(0.0), -st, F(0.0), ct;
        Rz << cpsi, -spsi, F(0.0), spsi, cpsi, F(0.0), F(0.0), F(0.0), F(1.0);

        return SE3(Rz * Ry * Rx, t);
    }

    static SE3 fromRPYAndVector(const Vec3F &rpy, const Vec3F &t)
    {
        return SE3::fromRPYAndVector(rpy(0), rpy(1), rpy(2), t);
    }

    static SE3 fromQuaternionAndVector(const Eigen::Matrix<F,4,1> &q, const Vec3F &t)
    {
        F qw{q(0)};
        Vec3F qv{q.template tail<3>()};
        Mat3F qv_x{skew3(qv)};

        Mat3F R{(2 * pow(qw,2) - 1.0) * Mat3F::Identity() + 2 * qw * qv_x + 2 * qv * qv.transpose()}; //Active rotation
        return SE3(R, t);
    }

    static SE3 fromQuaternionAndVector(const Eigen::Quaternion<F> &q, const Vec3F &t)
    {
        Mat3F R(q);
        return SE3(R, t);
    }

private:
    Mat4F _arr;
};

#endif