#ifndef RIGIDTRANSFORMATION_QUATERNION_H_
#define RIGIDTRANSFORMATION_QUATERNION_H_

#include <Eigen/Core>
#include <cmath>
#include <iostream>
#include <random>

// Remove so3 when I figure out the euler angle stuff
#include "so3.h"
#include "utils.h"

namespace rigidTransform {

template <typename T>
class Quaternion {
   public:
    using Vec4T = Eigen::Matrix<T, 4, 1>;
    using Mat3T = Eigen::Matrix<T, 3, 3>;
    using Vec3T = Eigen::Matrix<T, 3, 1>;

    Quaternion() : arr_(data_) {
        arr_.setZero();
        arr_(0) = T(1.0);
    }

    explicit Quaternion(const T *data) : arr_(const_cast<T *>(data)) {
        if (arr_(0) < T(0)) arr_ *= T(-1);
    }

    explicit Quaternion(const Eigen::Ref<const Vec4T> &q) : arr_(data_) {
        arr_ = q;
        if (arr_(0) < T(0)) arr_ *= T(-1);
    }

    Quaternion(const T &phi, const T &theta, const T &psi) : arr_(data_) {
        T cp{cos(phi / 2.0)}, sp{sin(phi / 2.0)};
        T ct{cos(theta / 2.0)}, st{sin(theta / 2.0)};
        T cps{cos(psi / 2.0)}, sps{sin(psi / 2.0)};

        Vec4T q{Vec4T::Zero()};
        q(0) = cp * ct * cps - sp * st * sps;
        q(1) = sp * ct * cps + cp * st * sps;
        q(2) = cp * st * cps - sp * ct * sps;
        q(3) = cp * ct * sps + sp * st * cps;

        if (q(0) < T(0.0)) q *= T(-1);

        arr_ = q;
    }

    Quaternion(const Quaternion &q) : arr_(data_) {
        arr_ = q.q();
        if (arr_(0) < T(0)) arr_ *= T(-1);
    }

    Quaternion &operator=(const Quaternion &rhs) {
        arr_ = rhs.q();
        return (*this);
    }

    template <typename T2>
    Quaternion operator*(const Quaternion<T2> &rhs) const {
        return this->template otimes<T, T2>(rhs);
    }

    template <typename T2>
    Quaternion operator*=(const Quaternion<T2> &rhs) {
        (*this) = this->template otimes<T, T2>(rhs);
        return (*this);
    }

    Vec4T q() const { return arr_; }

    T qw() const { return arr_(0); }
    T qx() const { return arr_(1); }
    T qy() const { return arr_(2); }
    T qz() const { return arr_(3); }
    Vec3T qv() const { return arr_.template tail<3>(); }

    Mat3T R() const {
        Vec3T qv = this->qv();
        Mat3T R = (2 * pow(qw(), 2) - 1) * Mat3T::Identity() +
                  2 * qw() * skew3<T>(qv) + 2 * qv * qv.transpose();
        return R;
    }

    // Not a very efficient way to do it but it works
    Vec3T euler() const {
        SO3<T> Mat = SO3<T>::fromQuat(arr_);
        return Mat.euler();
    }

    Mat3T Adj() const { return R(); }

    T norm() const { return arr_.norm(); }

    void normalize_() { arr_ = arr_ / norm(); }

    Quaternion inverse() const {
        Vec4T q;
        q << qw(), -qv();
        return Quaternion(q);
    }

    void inverse_() { arr_.template tail<3>() *= -1; }

    template <typename T2, typename Tout = T>
    Eigen::Matrix<Tout, 3, 1> rotate(const Eigen::Ref<const Eigen::Matrix<T2, 3, 1>> &v) const {
        T _qw = qw();
        Vec3T _qv = qv();

        Vec3T t(T(2) * skew3<T2>(v) * _qv);
        return v - _qw * t + skew3<T>(t) * _qv;
    }

    template <typename T2, typename Tout = T>
    Eigen::Matrix<Tout, 3, 1> inv_rotate(const Eigen::Ref<const Eigen::Matrix<T2, 3, 1>> &v) const {
        return inverse().template rotate<T2, Tout>(v);
    }

    template <typename T2, typename Tout = T>
    Quaternion<Tout> boxplusr(const Eigen::Ref<const Eigen::Matrix<T2, 3, 1>> &v) const {
        return (*this).template otimes<T2, Tout>(Quaternion::Exp(v));
    }

    template <typename T2, typename Tout = T>
    Eigen::Matrix<Tout, 3, 1> boxminusr(const Quaternion<T2> &q) const {
        return Quaternion<Tout>::Log(q.inverse().template otimes<T, Tout>(*this));
    }

    template <typename T2, typename Tout = T>
    Quaternion<Tout> boxplusl(const Eigen::Ref<const Eigen::Matrix<T2, 3, 1>> &v) const {
        return Quaternion::Exp(v).template otimes<T, Tout>(*this);
    }

    template <typename T2, typename Tout = T>
    Eigen::Matrix<Tout, 3, 1> boxminusl(const Quaternion<T2> &q) const {
        return Quaternion::Log((*this).template otimes<T2, Tout>(q.inverse()));
    }

    T *data() { return arr_.data(); }

    static Quaternion fromAxisAngle(const Eigen::Ref<const Vec3T> &v) {
        return Quaternion::Exp(v);
    }

    static Quaternion fromR(const Eigen::Ref<const Mat3T> &R) {
        T d{R.trace()};
        Vec4T q;
        if (d > 0) {
            T s{2 * sqrt(d + 1)};
            q << s / 4, (R(1, 2) - R(2, 1)) / s, (R(2, 0) - R(0, 2)) / s,
                (R(0, 1) - R(1, 0)) / s;
        } else if (R(0, 0) > R(1, 1) && R(0, 0) > R(2, 2)) {
            T s{2 * sqrt(1 + R(0, 0) - R(1, 1) - R(2, 2))};
            q << (R(1, 2) - R(2, 1)) / s, s / 4, (R(1, 0) + R(0, 1)) / s,
                (R(2, 0) + R(0, 2)) / s;
        } else if (R(1, 1) > R(2, 2)) {
            T s{2 * sqrt(1 + R(1, 1) - R(0, 0) - R(2, 2))};
            q << (R(2, 0) - R(0, 2)) / s, (R(1, 0) + R(0, 1)) / s, s / 4,
                (R(2, 1) + R(1, 2)) / s;
        } else {
            T s{2 * sqrt(1 + R(2, 2) - R(0, 0) - R(1, 1))};
            q << (R(0, 1) - R(1, 0)) / s, (R(2, 0) + R(0, 2)) / s,
                (R(2, 1) + R(1, 2)) / s, s / 4;
        }
        q.template tail<3>() *= -1;

        if (q(0) < T(0.0)) q *= -1;

        return Quaternion(q);
    }

    static Quaternion random() {
        Vec3T u;
        u << randomScalar<T>(0, 1), randomScalar<T>(0, 1),
            randomScalar<T>(0, 1);
        Vec4T q;
        q << sin(2 * PI * u(1)) * sqrt(1 - u(0)),
            cos(2 * PI * u(1)) * sqrt(1 - u(0)),
            sin(2 * PI * u(2)) * sqrt(u(0)), cos(2 * PI * u(2)) * sqrt(u(0));

        if (q(0) < 0) q *= -1;
        return Quaternion(q);
    }

    static Quaternion Identity() { return Quaternion(); }

    static Quaternion Exp(const Eigen::Ref<const Vec3T> &v) {
        T theta{v.norm()};
        Vec3T vec{v / theta};
        Vec4T q{Vec4T::Zero()};
        if (abs(theta) > 1e-4) {
            q(0) = cos(theta / T(2.0));
            q.template tail<3>() = vec * sin(theta / T(2.0));
        } else {
            q(0) = T(1.0);
            q.template tail<3>() = v / T(2.0);
            q = q / q.norm();
        }

        if (q(0) < T(0)) q *= T(-1);

        return Quaternion(q);
    }

    Vec3T Log() const { return Quaternion::Log(*this); }

    static Vec3T Log(const Quaternion &q) {
        T _qw(q.qw());
        Vec3T _qv(q.qv());
        T theta{_qv.norm()};

        Vec3T logq;
        if (abs(theta) > T(1e-8))
            logq = T(2) * atan(theta / _qw) * _qv / theta;
        else
            logq = T(2) * _qv / _qw;
        return logq;
    }

    template <typename T2, typename Tout = T>
    Quaternion<Tout> otimes(const Quaternion<T2> &R) const {
        Eigen::Matrix<T, 4, 4> Q;
        Q(0, 0) = qw();
        Q.template block<1, 3>(0, 1) = -qv().transpose();
        Q.template block<3, 1>(1, 0) = qv();
        Q.template block<3, 3>(1, 1) =
            qw() * Mat3T::Identity() + skew3<T>(qv());
        Eigen::Matrix<Tout, 4, 1> qp = Q * R.q();

        if (qp(0) < Tout(0)) qp *= Tout(-1);

        return Quaternion<Tout>(qp);
    }

   private:
    T data_[4];

   public:
    Eigen::Map<Vec4T> arr_;
};

template <typename T>
std::ostream &operator<<(std::ostream &os, const Quaternion<T> &q) {
    os << q.q().transpose();
    return os;
}
}  // namespace rigidTransform

#endif  // RIGIDTRANSFORMATION_QUATERNION_H_
