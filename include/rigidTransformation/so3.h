#ifndef RIGIDTRANSFORMATION_SO3_H_
#define RIGIDTRANSFORMATION_SO3_H_

#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <random>

#include "utils.h"

namespace rigidTransform {

template <typename T>
class SO3 {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Mat3T = Eigen::Matrix<T, 3, 3>;
    using Vec3T = Eigen::Matrix<T, 3, 1>;
    using Map3T = Eigen::Map<Mat3T>;

    SO3() : arr_(data_) { arr_.setIdentity(); }

    explicit SO3(const T *data) : arr_(const_cast<T *>(data)) {}

    explicit SO3(const Eigen::Ref<const Mat3T> &R) : arr_(data_) { arr_ = R; }

    SO3(const SO3 &R) : arr_(data_) { arr_ = R.R(); }

    SO3(T phi, T theta, T psi) : arr_(data_) {
        T cphi{cos(phi)}, sphi(sin(phi));
        T ct{cos(theta)}, st{sin(theta)};
        T cpsi{cos(psi)}, spsi{sin(psi)};

        Mat3T Rpsi, Rtheta, Rphi;
        Rpsi << cpsi, -spsi, 0, spsi, cpsi, 0, 0, 0, 1;
        Rtheta << ct, 0, st, 0, 1, 0, -st, 0, ct;
        Rphi << 1, 0, 0, 0, cphi, -sphi, 0, sphi, cphi;

        arr_ = Rphi * Rtheta * Rpsi;
    }

    Mat3T R() const { return arr_; }

    Vec3T euler() const {
        Mat3T R1 = R();
        T phi, theta, psi;
        // if (1 - abs(R1(2, 0)) > 1e-8) {
        //     phi = atan2(R1(2, 1), R1(2, 2));
        //     theta = asin(-R1(2, 0));
        //     psi = atan2(R1(1, 0), R1(0, 0));
        // } else {
        //     phi = 0.0;
        //     if (R1(2, 0) > 0.0) {
        //         theta = PI / 2.0;
        //         psi = -atan2(-R1(1, 2), R1(1, 1));
        //     } else {
        //         theta = -PI / 2.0;
        //         psi = atan2(-R1(1, 2), R1(1, 1));
        //     }
        // }
        phi = atan2(-R1(1, 2), R1(2, 2));
        theta = asin(R1(0, 2));
        psi = atan2(-R1(0, 1), R1(0, 0));

        return (Vec3T() << phi, theta, psi).finished();
    }

    Mat3T Adj() const { return arr_; }

    SO3 &operator=(const SO3 &rhs) {
        arr_ = rhs.R();
        return (*this);
    }

    template <typename T2>
    SO3 operator*(const SO3<T2> &rhs) const {
        return this->template otimes<T, T2>(rhs);
    }

    template <typename T2>
    SO3 &operator*=(const SO3<T2> &rhs) {
        (*this) = this->template otimes<T, T2>(rhs);
        return (*this);
    }

    SO3 inverse() const {
        SO3 temp;
        temp.arr_ = R().transpose();
        return temp;
    }

    void inverse_() { arr_.transposeInPlace(); }

    template <typename T2, typename Tout = T>
    Eigen::Matrix<Tout, 3, 1> rotate(const Eigen::Ref<const Eigen::Matrix<T2, 3, 1>> &v) const {
        return R() * v;
    }

    template <typename T2, typename Tout = T>
    Eigen::Matrix<Tout, 3, 1> inv_rotate(const Eigen::Ref<const Eigen::Matrix<T2, 3, 1>> &v) const {
        return inverse().R() * v;
    }

    template <typename T2, typename Tout = T>
    SO3<Tout> boxplusr(const Eigen::Ref<const Eigen::Matrix<T2, 3, 1>> &v) {
        return this->template otimes<T2, Tout>(SO3<T2>::Exp(v));
    }

    template <typename T2, typename Tout = T>
    Eigen::Matrix<Tout, 3, 1> boxminusr(const SO3<T2> &R2) const {
        return SO3<Tout>::Log(R2.inverse().template otimes<T, Tout>(*this));
    }

    template <typename T2, typename Tout = T>
    SO3<Tout> boxplusl(const Eigen::Ref<const Eigen::Matrix<T2, 3, 1>> &v) {
        return SO3<T2>::Exp(v).template otimes<T, Tout>(*this);
    }

    template <typename T2, typename Tout = T>
    Eigen::Matrix<Tout, 3, 1> boxminusl(const SO3<T2> &R2) const {
        return SO3<Tout>::Log(this->template otimes<T2, Tout>(R2.inverse()));
    }

    T *data() { return arr_.data(); }

    static SO3 fromAxisAngle(const Eigen::Ref<const Vec3T> &v) {
        T theta = v.norm();
        Mat3T vx{skew3<T>(v)};

        T A{1.0}, B{0.5};
        if (abs(theta) > 1e-8) {
            A = sin(theta) / theta;
            B = (1 - cos(theta)) / (theta * theta);
        }

        return SO3(Mat3T::Identity() + A * vx + B * vx * vx);
    }

    static SO3 fromQuat(const Eigen::Ref<const Eigen::Matrix<T, 4, 1>> &q) {
        T qw = q(0);
        Vec3T qv = q.template tail<3>();
        Mat3T R = Mat3T::Identity() * (2 * qw * qw - 1);
        R = R + 2 * qw * skew3<T>(qv) + 2 * qv * qv.transpose();
        return SO3(R);
    }

    static SO3 random() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dist(0, 1);

        Vec3T x;
        x << dist(gen), dist(gen), dist(gen);
        T psi{2 * PI * x(0)};
        Mat3T Rpsi;
        Rpsi << cos(psi), sin(psi), 0, -sin(psi), cos(psi), 0, 0, 0, 1;

        Vec3T v;
        v << cos(2 * PI * x(1)) * sqrt(x(2)), sin(2 * PI * x(1)) * sqrt(x(2)),
            sqrt(1 - x(2));
        Mat3T H = Mat3T::Identity() - 2 * v * v.transpose();
        Mat3T res = -H * Rpsi;
        return SO3(res);
    }

    static SO3 Identity() { return SO3(); }

    Vec3T Log() const { return SO3::Log(*this); }

    static Vec3T Log(const SO3 &R) {
        T theta = acos((R.R().trace() - T(1)) / 2.0);
        Mat3T logR{Mat3T::Zero()};
        if (abs(theta) < 1e-8) {
            logR = T(0.5) * (R.R() - R.R().transpose());
        } else if (abs(abs(theta) - PI) < 1e-8) {
            // Find a way to simplify this
            // Do angle Pi - theta around the negative axis??
        } else {
            logR = theta / (T(2) * sin(theta)) * (R.R() - R.R().transpose());
        }

        return SO3::vee(logR);
    }

    static SO3 Exp(const Eigen::Ref<const Vec3T> &logR) {
        Mat3T logRx{skew3<T>(logR)};
        T theta{logR.norm()};
        Mat3T R{Mat3T::Identity()};
        if (abs(theta) > 1e-8) {
            R = Mat3T::Identity() + sin(theta) / theta * logRx +
                (T(1) - cos(theta)) / pow(theta, 2) * logRx * logRx;
        } else {
            R = R + logRx;
        }
        return SO3(R);
    }

    static Vec3T vee(const Eigen::Ref<const Mat3T> &M) {
        Vec3T v;
        v << M(2, 1), M(0, 2), M(1, 0);
        return v;
    }

    template <typename T2, typename Tout = T>
    SO3<Tout> otimes(const SO3<T2> &R) const {
        SO3<Tout> res;
        res.arr_ = arr_ * R.R();
        return res;
    }

   private:
    T data_[9];

   public:
    Eigen::Map<Mat3T> arr_;
};

template <typename T>
std::ostream &operator<<(std::ostream &os, const SO3<T> &R) {
    os << R.R();
    return os;
}
}  // namespace rigidTransform

#endif  // RIGIDTRANSFORMATION_SO3_H_
