#ifndef RIGIDTRANSFORMATION_SE2_H_
#define RIGIDTRANSFORMATION_SE2_H_

#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <random>

#include "utils.h"

namespace rigidTransform {

template <typename F>
class SE2 {
    using Mat3F = Eigen::Matrix<F, 3, 3>;
    using Mat2F = Eigen::Matrix<F, 2, 2>;
    using Vec2F = Eigen::Matrix<F, 2, 1>;
    using Vec3F = Eigen::Matrix<F, 3, 1>;
    using Map3F = Eigen::Map<Mat3F>;

   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    SE2() : arr_(data_) { arr_.setIdentity(); }

    explicit SE2(const F *data) : arr_(const_cast<F *>(data)) {}

    explicit SE2(const Eigen::Ref<const Mat3F> &T) : arr_(data_) { arr_ = T; }

    SE2(const Eigen::Ref<const Vec2F> &t, const F &theta) : arr_(data_) {
        F ct{cos(theta)}, st{sin(theta)};
        arr_ << ct, -st, t(0), st, ct, t(1), F(0), F(0), F(1);
    }

    SE2(const F &x, const F &y, const F &theta) : arr_(data_) {
        F ct{cos(theta)}, st{sin(theta)};
        arr_ << ct, -st, x, st, ct, y, 0, 0, 1;
    }

    SE2(const Eigen::Ref<const Mat2F> &R, const Eigen::Ref<const Vec2F> &t)
        : arr_(data_) {
        arr_.setIdentity();
        arr_.template block<2, 2>(0, 0) = R;
        arr_.template block<2, 1>(0, 2) = t;
    }

    SE2(const Eigen::Ref<const Mat2F> &R, const F &x, const F &y)
        : arr_(data_) {
        arr_.setIdentity();
        arr_.template block<2, 2>(0, 0) = R;
        arr_(0, 2) = x;
        arr_(1, 2) = y;
    }

    SE2(const SE2 &T) : arr_(data_) { arr_ = T.T(); }

    SE2 &operator=(const SE2 &rhs) {
        arr_ = rhs.T();
        return (*this);
    }

    template <typename F2>
    SE2 operator*(const SE2<F2> &rhs) const {
        return this->template otimes<F, F2>(rhs);
    }

    Mat3F T() const { return arr_; }

    Mat2F R() const { return arr_.template block<2, 2>(0, 0); }

    Vec2F t() const { return arr_.template block<2, 1>(0, 2); }

    F theta() const { return SE2::Log(*this)(2); }
    F x() const { return arr_(0, 2); }
    F y() const { return arr_(1, 2); }

    Mat3F Adj() const {
        Mat2F J;
        J << 0, -1, 1, 0;
        Mat3F adj(Mat3F::Identity());
        adj.template block<2, 2>(0, 0) = R();
        adj.template block<2, 1>(0, 2) = -J * t();

        return adj;
    }

    SE2 inverse() const {
        Mat3F T(Mat3F::Identity());
        T.template block<2, 2>(0, 0) = R().transpose();
        T.template block<2, 1>(0, 2) = -R().transpose() * t();
        return SE2(T);
    }

    void inverse_() {
        arr_.template block<2, 2>(0, 0) = R().transpose();
        arr_.template block<2, 1>(0, 2) = -R() * t();
    }

    template <typename F2, typename Fout = F>
    Eigen::Matrix<Fout, 2, 1> transform(const Eigen::Ref<const Eigen::Matrix<F, 2, 1>> &pt) const {
        return R() * pt + t();
    }

    template <typename F2, typename Fout = F>
    Eigen::Matrix<Fout, 2, 1> inv_transform(const Eigen::Ref<const Eigen::Matrix<F, 2, 1>> &pt) const {
        return inverse().template transform<F2, Fout>(pt);
    }

    template <typename F2, typename Fout = F>
    SE2<Fout> boxplusr(const Eigen::Ref<const Eigen::Matrix<F2, 3, 1>> &tau) const {
        return this->template otimes<F2, Fout>(SE2::Exp(tau));
    }

    template <typename F2, typename Fout = F>
    Eigen::Matrix<Fout, 3, 1> boxminusr(const SE2<F2> &T) const {
        return SE2<Fout>::Log(T.inverse().template otimes<F, Fout>(*this));
    }

    template <typename F2, typename Fout = F>
    SE2<Fout> boxplusl(const Eigen::Ref<const Eigen::Matrix<F2, 3, 1>> &tau) const {
        return SE2<F2>::Exp(tau).template otimes<F, Fout>(*this);
    }

    template <typename F2, typename Fout = F>
    Eigen::Matrix<Fout, 3, 1> boxminusl(const SE2<F2> &T) const {
        return SE2<Fout>::Log(this->template otimes<F2, Fout>(T.inverse()));
    }

    F *data() { return arr_.data(); }

    static SE2 random() {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_real_distribution<F> r_dist(-PI, PI);
        static std::uniform_real_distribution<F> t_dist(-10.0, 10.0);

        F theta{r_dist(gen)};
        F x{t_dist(gen)}, y{t_dist(gen)};
        return SE2(x, y, theta);
    }

    static SE2 Identity() { return SE2(); }

    Vec3F Log() const { return SE2::Log((*this)); }

    static Vec3F Log(const SE2 &T) {
        F theta{atan2(T.T()(1, 0), T.T()(0, 0))};  // should I add () to index??
        Vec2F t(T.t());

        F A, B;
        if (abs(theta) > F(1e-8)) {
            A = sin(theta) / theta;
            B = (F(1.0) - cos(theta)) / theta;
        } else {
            A = F(1.0);
            B = theta / F(2.0);
        }
        F normalizer{F(1.0) / (A * A + B * B)};
        Mat2F temp_arr;
        temp_arr << A, B, -B, A;
        Mat2F V_inv = temp_arr * normalizer;

        Vec3F logT;
        logT << V_inv * t, theta;
        return logT;
    }

    static SE2 Exp(const Eigen::Ref<const Vec3F> &logT) {
        F theta(logT(2));
        Vec2F rho(logT.template head<2>());

        F A, B;
        if (abs(theta) > F(1e-8)) {
            A = sin(theta) / theta;
            B = (F(1) - cos(theta)) / theta;
        } else {
            A = F(1.0);
            B = theta / F(2.0);
        }
        Mat2F V;
        V << A, -B, B, A;
        Vec2F t(V * rho);
        return SE2(t, theta);
    }

    template <typename F2, typename Fout = F>
    SE2<Fout> otimes(const SE2<F2> &rhs) const {
        return SE2<Fout>(this->T() * rhs.T());
    }

   private:
    F data_[9];

   public:
    Eigen::Map<Mat3F> arr_;
};

template <typename F>
std::ostream &operator<<(std::ostream &os, const SE2<F> &T) {
    os << T.T();
    return os;
}

}  // namespace rigidTransform

#endif  // RIGIDTRANSFORMATION_SE2_H_
