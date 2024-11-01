#ifndef RIGIDTRANSFORMATION_SO2_H_
#define RIGIDTRANSFORMATION_SO2_H_
#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <random>

#include "utils.h"

namespace rigidTransform {

template <typename T>
class SO2 {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Mat2T = Eigen::Matrix<T, 2, 2>;
    using Vec2T = Eigen::Matrix<T, 2, 1>;
    using Map2T = Eigen::Map<Mat2T>;

    SO2() : arr_(data_) {
        arr_.setIdentity();
    }

    explicit SO2(const T *data) : arr_(const_cast<T *>(data)) {}

    explicit SO2(const Eigen::Ref<const Mat2T> &R) : arr_(data_) { arr_ = R; }

    explicit SO2(const T &ang) : arr_(data_) {
        T ct{cos(ang)}, st{sin(ang)};
        arr_ << ct, -st, st, ct;
    }

    SO2(const SO2 &R) : arr_(data_) {
        arr_ = R.R();
    }

    SO2 &operator=(const SO2 &rhs) {
        arr_ = rhs.R();
        return *this;
    }

    template <typename T2>
    SO2 operator*(const SO2<T2> &rhs) const {
        SO2 temp = this->template otimes<T, T2>(rhs);
        return temp;
    }

    template <typename T2>
    SO2 &operator*=(const SO2<T2> &rhs) {
        (*this) = this->template otimes<T, T2>(rhs);
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

    template <typename Tout = T, typename T2>
    Eigen::Matrix<Tout, 2, 1> rotate(const Eigen::Ref<const Eigen::Matrix<T2, 2, 1>> &v) const {
        return R() * v;
    }

    template <typename Tout = T, typename T2>
    Eigen::Matrix<Tout, 2, 1> inv_rotate(const Eigen::Ref<const Eigen::Matrix<T2, 2, 1>> &v) const {
        return inverse().R() * v;
    }

    T Log() const {
        return SO2::Log(*this);
    }

    T Adj() const {
        return 1.0;
    }

    template <typename Tout = T, typename T2>
    SO2<Tout> boxplusr(const T2 &ang) const {
        return otimes<Tout, T2>(SO2::Exp(ang));
    }

    template <typename Tout = T, typename T2>
    Tout boxminusr(const SO2<T2> &R) const {
        return SO2<Tout>::Log(R.inverse().template otimes<Tout, T>(*this));
    }

    template <typename Tout = T, typename T2>
    SO2<Tout> boxplusl(const T2 &ang) const {
        return SO2::Exp(ang).template otimes<T, T>(*this);
    }

    template <typename Tout = T, typename T2>
    Tout boxminusl(const SO2<T2> &R) const {
        return SO2<Tout>::Log(this->template otimes<Tout, T2>(R.inverse()));
    }

    T *data() {
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

    template <typename Tout = T, typename T2>
    SO2<Tout> otimes(const SO2<T2> &R) const {
        SO2<Tout> res;
        res.arr_ = arr_ * R.R();
        return res;
    }

   private:
    T data_[4];

    Eigen::Map<Mat2T> arr_;
};

template <typename T>
std::ostream &operator<<(std::ostream &os, const SO2<T> &R) {
    os << R.R();
    return os;
}

}  // namespace rigidTransform

#endif  // RIGIDTRANSFORMATION_SO2_H_
