#ifndef SE2_H
#define SE2_H

#include <Eigen/Dense>
#include <random>
#include <iostream>
#include <cmath>

#include "utils.h"

namespace rigidTransform
{

template<typename F>
class SE2
{
    using Mat3F = Eigen::Matrix<F,3,3>;
    using Mat2F = Eigen::Matrix<F,2,2>;
    using Vec2F = Eigen::Matrix<F,2,1>;
    using Vec3F = Eigen::Matrix<F,3,1>;
    using Map3F = Eigen::Map<Mat3F>;
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    SE2() : arr_(data_) { arr_.setIdentity(); }

    SE2(const F *data) : arr_(const_cast<F*>(data)) {}

    SE2(const Eigen::Ref<const Mat3F> &T) : arr_(data_) { arr_ = T; }

    SE2(const Eigen::Ref<const Vec2F> &t, const F& theta) : arr_(data_)
    {
        F ct{cos(theta)}, st{sin(theta)};
        arr_ << ct, -st, t(0), st, ct, t(1), 0, 0, 1;
    }

    SE2(const F& x, const F& y, const F& theta) : arr_(data_)
    {
        F ct{cos(theta)}, st{sin(theta)};
        arr_ << ct, -st, x, st, ct, y, 0, 0, 1;
    }

    Mat3F T() const { return arr_; }

private:
    F data_[9];
public:
    Eigen::Map<Mat3F> arr_;
};

} // namespace rigidTransform

#endif
