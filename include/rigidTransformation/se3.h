#ifndef SE3_H
#define SE3_H

#include <random>
#include <iostream>
#include <cmath>

#include <eigen3/Eigen/Dense>

#include "utils.h"
#include "quaternion.h"

namespace rigidTransform
{

template<typename F>
class SE3
{
    using Mat3F = Eigen::Matrix<F,3,3>;
    using Vec7F = Eigen::Matrix<F,7,1>;
    using Vec6F = Eigen::Matrix<F,6,1>;
    using Vec4F = Eigen::Matrix<F,4,1>;
    using Vec3F = Eigen::Matrix<F,3,1>;
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    SE3() : arr_(data_), q_(data_+3)
    {
        arr_.setZero();
        arr_(3) = 1.0;
    }

    SE3(const F* data) : arr_(const_cast<F*>(data)), q_(const_cast<F*>(data+3)) {}

    SE3(const Eigen::Ref<const Vec7F> &T) : arr_(data_), q_(data_+3)
    {
        arr_ = T;
    }

    SE3(const F& r, const F& p, const F& y, const Eigen::Ref<const Vec3F> &t) : arr_(data_), q_(data_+3)
    {
        arr_.template head<3>() = t;
        arr_.template tail<4>() = Quaternion<F>(r,p,y).q();
    }

    SE3(const Eigen::Ref<const Mat3F> &R, const Eigen::Ref<const Vec3F> &t): arr_(data_), q_(data_+3)
    {
        arr_.template head<3>() = t;
        arr_.template tail<4>() = Quaternion<F>::fromR(R).q();
    }

    SE3(const Quaternion<F> &q, const Eigen::Ref<const Vec3F> &t) : arr_(data_), q_(data_+3)
    {
        arr_.template head<3>() = t;
        arr_.template tail<4>() = q.q();
    }

    SE3(const SE3 &T): arr_(data_), q_(data_+3)
    {
        arr_ = T.T();
    }

    SE3& operator=(const SE3& rhs)
    {
        arr_ = rhs.T();
        return (*this);
    }

    SE3 operator*(const SE3& rhs)
    {
        Quaternion<F> q(quat() * rhs.quat());
        Vec3F trans(t() + quat().rota(rhs.t()));
        return SE3(q,trans);
    }

    Vec7F T() const { return arr_; }
    Mat3F R() const { return q_.R(); }
    Vec4F q() const { return q_.q(); }
    Quaternion<F> quat() const { return q_; } // better name for this
    Vec3F t() const { return arr_.template head<3>(); }

    SE3 inverse() const
    {
        Quaternion<F> q_inv = q_.inverse();
        Vec3F t_inv = -q_inv.rota(t());
        return SE3(q_inv, t_inv);
    }

    void inverse_()
    {
        q_.inverse_();
        arr_.template head<3>() = -q_.rota(t());
    }

    Vec3F transa(const Eigen::Ref<const Vec3F> &v)
    {
        return t() + q_.rota(v);
    }

    Vec3F transp(const Eigen::Ref<const Vec3F> &v)
    {
        return inverse().transa(v);
    }

    static SE3 fromAxisAngleAndt(const Eigen::Ref<const Vec3F> &v, const Eigen::Ref<const Vec3F> &t)
    {
        Quaternion<F> q(Quaternion<F>::fromAxisAngle(v));
        return SE3(q, t);
    }

    static SE3 random()
    {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_real_distribution<F> dist(-10, 10);

        Quaternion<F> q(Quaternion<F>::random());
        Vec3F t;
        t << dist(gen), dist(gen), dist(gen);

        return SE3(q, t);
    }

    static SE3 Identity()
    {
        return SE3();
    }
private:
    F data_[7];
    Eigen::Map<Vec7F> arr_;
    Quaternion<F> q_;
};

template<typename F>
std::ostream& operator <<(std::ostream &os, const SE3<F> &T)
{
    os << T.T().transpose();
    return os;
}
} // namespace rigidTransform

#endif
