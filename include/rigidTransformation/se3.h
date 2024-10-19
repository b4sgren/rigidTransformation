#ifndef RIGIDTRANSFORMATION_SE3_H_
#define RIGIDTRANSFORMATION_SE3_H_

#include <cmath>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <random>

#include "quaternion.h"
#include "utils.h"

namespace rigidTransform {

template <typename F>
class SE3 {
    using Mat3F = Eigen::Matrix<F, 3, 3>;
    using Vec7F = Eigen::Matrix<F, 7, 1>;
    using Vec6F = Eigen::Matrix<F, 6, 1>;
    using Vec4F = Eigen::Matrix<F, 4, 1>;
    using Vec3F = Eigen::Matrix<F, 3, 1>;

   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    SE3() : arr_(data_), q_(data_ + 3) {
        arr_.setZero();
        arr_(3) = F(1.0);
    }

    explicit SE3(const F *data)
        : arr_(const_cast<F *>(data)), q_(const_cast<F *>(data + 3)) {}

    explicit SE3(const Eigen::Ref<const Vec7F> &T)
        : arr_(data_), q_(data_ + 3) {
        arr_ = T;
    }

    // explicit SE3(const Eigen::Ref<const Vec7F> &T)
    //     : arr_(const_cast<F *>(T.data())), q_(arr_.data() + 3) {
    //     arr_ = T;
    // }

    SE3(const F &r, const F &p, const F &y, const Eigen::Ref<const Vec3F> &t)
        : arr_(data_), q_(data_ + 3) {
        arr_.template head<3>() = t;
        arr_.template tail<4>() = Quaternion<F>(r, p, y).q();
    }

    SE3(const F &x, const F &y, const F &z, const F &qw, const F &qx, const F &qy, const F &qz) : arr_(data_), q_(data_ + 3) {
        arr_(0) = x;
        arr_(1) = y;
        arr_(2) = z;
        arr_(3) = qw;
        arr_(4) = qx;
        arr_(5) = qy;
        arr_(6) = qz;
    }

    SE3(const F &x, const F &y, const F &z, const F &r, const F &p, const F &h)
        : SE3(r, p, h, Vec3F(x, y, z)) {}

    SE3(const Eigen::Ref<const Mat3F> &R, const Eigen::Ref<const Vec3F> &t)
        : arr_(data_), q_(data_ + 3) {
        arr_.template head<3>() = t;
        arr_.template tail<4>() = Quaternion<F>::fromR(R).q();
    }

    SE3(const Quaternion<F> &q, const Eigen::Ref<const Vec3F> &t)
        : arr_(data_), q_(data_ + 3) {
        arr_.template head<3>() = t;
        arr_.template tail<4>() = q.q();
    }

    SE3(const SE3 &T) : arr_(data_), q_(data_ + 3) { arr_ = T.T(); }

    SE3 &operator=(const SE3 &rhs) {
        arr_ = rhs.T();
        return (*this);
    }

    template <typename F2>
    SE3 operator*(const SE3<F2> &T2) const {
        Quaternion<F> q(q_ * T2.q_);
        Vec3F trans(t() + q_.template rotate<F2>(T2.t()));
        return SE3(q, trans);
    }

    Vec7F T() const { return arr_; }
    Mat3F R() const { return q_.R(); }
    Vec4F q() const { return q_.q(); }
    Quaternion<F> quat() const { return q_; }
    Vec3F t() const { return arr_.template head<3>(); }
    F x() const { return arr_(0); }
    F y() const { return arr_(1); }
    F z() const { return arr_(2); }
    Vec3F euler() const { return q_.euler(); }
    Eigen::Matrix<F, 4, 4> matrix() const {
        Eigen::Matrix<F, 4, 4> mat = Eigen::Matrix<F, 4, 4>::Identity();
        mat.template block<3, 3>(0, 0) = R();
        mat.template block<3, 1>(0, 3) = t();
        return mat;
    }

    SE3 inverse() const {
        Quaternion<F> q_inv = q_.inverse();
        Vec3F t_inv = -q_inv.template rotate<F>(t());
        return SE3(q_inv, t_inv);
    }

    void inverse_() {
        q_.inverse_();
        arr_.template head<3>() = -q_.template rotate<F>(t());
    }

    template <typename F2>
    Vec3F transform(const Eigen::Ref<const Eigen::Matrix<F2, 3, 1>> &v) const {
        return t() + q_.template rotate<F>(v);
    }

    template <typename F2>
    Vec3F inv_transform(const Eigen::Ref<const Eigen::Matrix<F2, 3, 1>> &v) const {
        return inverse().transform(v);
    }

    template <typename F2>
    SE3 boxplusr(const Eigen::Ref<const Eigen::Matrix<F2, 6, 1>> &tau) const {
        return (*this) * SE3::Exp(tau);
    }

    Vec6F boxminusr(const SE3 &T) const {
        return SE3::Log(T.inverse() * (*this));
    }

    template <typename F2>
    SE3 boxplusl(const Eigen::Ref<const Eigen::Matrix<F2, 6, 1>> &tau) const {
        return SE3::Exp(tau) * (*this);
    }

    Vec6F boxminusl(const SE3 &T) const {
        return SE3::Log((*this) * T.inverse());
    }

    Eigen::Matrix<F, 6, 6> Adj() const {
        Mat3F rot = R();
        Eigen::Matrix<F, 6, 6> adj;
        adj.setZero();
        adj.template block<3, 3>(0, 0) = rot;
        adj.template block<3, 3>(3, 3) = rot;
        adj.template block<3, 3>(0, 3) = skew3<F>(t()) * rot;
        return adj;
    }

    F *data() { return arr_.data(); }

    static SE3 fromAxisAngleAndt(const Eigen::Ref<const Vec3F> &v,
                                 const Eigen::Ref<const Vec3F> &t) {
        Quaternion<F> q(Quaternion<F>::fromAxisAngle(v));
        return SE3(q, t);
    }

    static SE3 random() {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_real_distribution<F> dist(-10, 10);

        Quaternion<F> q(Quaternion<F>::random());
        Vec3F t;
        t << dist(gen), dist(gen), dist(gen);

        return SE3(q, t);
    }

    static SE3 Identity() { return SE3(); }

    static SE3 Exp(const Eigen::Ref<const Vec6F> &tau) {
        auto rho(tau.template head<3>());
        auto theta(tau.template tail<3>());
        F norm(theta.norm());

        Quaternion<F> q(Quaternion<F>::Exp(theta));

        Vec3F t;
        if (abs(norm) > F(1e-8)) {
            Mat3F thetax(skew3<F>(theta));
            Mat3F V = Mat3F::Identity() +
                      (F(1) - cos(norm)) / (norm * norm) * thetax +
                      (norm - sin(norm)) / pow(norm, 3) * thetax * thetax;
            t = V * rho;
        } else {
            t = rho;
        }

        return SE3(q, t);
    }

    Vec6F Log() const { return SE3::Log(*this); }

    static Vec6F Log(const SE3 &T) {
        Vec3F logq(Quaternion<F>::Log(T.q_));
        F theta(logq.norm());

        Vec3F rho;
        if (abs(theta) > F(1e-8)) {
            F A(sin(theta) / theta), B((F(1) - cos(theta)) / (theta * theta));
            Mat3F thetax(skew3<F>(logq));
            Mat3F V_inv(Mat3F::Identity() - F(0.5) * thetax +
                        (F(1) - A / (F(2) * B)) / (theta * theta) * thetax *
                            thetax);
            rho = V_inv * T.t();
        } else {
            rho = T.t();
        }

        Vec6F logT;
        logT << rho, logq;
        return logT;
    }

   private:
    F data_[7];

   public:
    Eigen::Map<Vec7F> arr_;
    Quaternion<F> q_;
};

template <typename F>
std::ostream &operator<<(std::ostream &os, const SE3<F> &T) {
    os << T.T().transpose();
    return os;
}
}  // namespace rigidTransform

#endif  // RIGIDTRANSFORMATION_SE3_H_
