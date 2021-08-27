#ifndef RIGIDTRANSFORMATION_SO2_H_
#define RIGIDTRANSFORMATION_SO2_H_
#include <Eigen/Dense>
#include <iostream>
#include <random>
#include <cmath>

#include "utils.h"

namespace rigidTransform {

template<typename T>
class SO2 {
 public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Mat2T = Eigen::Matrix<T, 2, 2>;
    using Vec2T = Eigen::Matrix<T, 2, 1>;
    using Map2T = Eigen::Map<Mat2T>;

    SO2() : arr_(data_) {
        arr_.setIdentity();
    }

    explicit SO2(const T *data) : arr_(const_cast<T*>(data)) {}

    explicit SO2(const Eigen::Ref<const Mat2T> &R) : arr_(data_) { arr_ = R; }

    explicit SO2(const T& ang): arr_(data_) {
        T ct{cos(ang)}, st{sin(ang)};
        arr_ << ct, -st, st, ct;
    }

    SO2(const SO2 &R): arr_(data_) {
        arr_ = R.R();
    }

    SO2& operator=(const SO2 &rhs) {
        arr_ = rhs.R();
        return *this;
    }

    // What type to return??
    template <typename T2>
    SO2 operator*(const SO2<T2> &rhs) const {
        SO2 temp;
        temp.arr_ = R() * rhs.R();
        return temp;
    }

    template <typename T2>
    SO2& operator*=(const SO2<T2> &rhs) {
        (*this) = (*this) * rhs;
        return (*this);
    }

    T operator()(int i, int j) const {
        return arr_(i, j);
    }

    Mat2T R() const { return arr_; }

    T theta() const { return SO2::Log(*this); }

    T det() const {
        return arr_.determinant();
    }

    SO2 inverse() const {
        SO2 temp;
        temp.arr_ = R().transpose();
        return temp;
    }

    // operations with _ at the end are in place
    void inverse_() {
        arr_.transposeInPlace();
    }

    Vec2T rota(const Eigen::Ref<const Vec2T> &v) const {
        return R() * v;
    }

    Vec2T rotp(const Eigen::Ref<Vec2T> &v) const {
        return inverse().R() * v;
    }

    T Log() const {
        return SO2::Log(*this);
    }

    T Adj() const {
        return 1.0;
    }

    template <typename T2>
    SO2 boxplusr(const T2 &ang) const {
        return (*this) * SO2::Exp(ang);
    }

    template <typename T2>
    T boxminusr(const SO2<T2> &R) const {
        return SO2::Log(R.inverse() * (*this));
    }

    template <typename T2>
    SO2 boxplusl(const T2 &ang) const {
        return SO2::Exp(ang) * (*this);
    }

    template <typename T2>
    T boxminusl(const SO2<T2> &R) const {
        return SO2::Log((*this) * R.inverse());
    }

    T* data() {
        return arr_.data();
    }

    static SO2 random() {
        T ang{randomScalar(T(-PI), T(PI))};
        return SO2(ang);
    }

    static SO2 identity() {
        return SO2();
    }

    static T Log(const SO2 &R) {
        return atan2(R(1, 0), R(0, 0));
    }

    static SO2 Exp(const T &ang) {
        return SO2(ang);
    }

 private:
    T data_[4];

//  public:
    Eigen::Map<Mat2T> arr_;
};

template<typename T>
std::ostream& operator <<(std::ostream &os, const SO2<T> &R) {
    os << R.R();
    return os;
}

}  // namespace rigidTransform

#endif  // RIGIDTRANSFORMATION_SO2_H_
