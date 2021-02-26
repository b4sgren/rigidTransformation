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

    Vec7F T() const { return arr_; }
    Vec4F q() const { return q_.q(); }
    Vec3F t() const { return arr_.template head<3>(); }
private:
    F data_[7];
    Eigen::Map<Vec7F> arr_;
    Quaternion<F> q_;
};

} // namespace rigidTransform

#endif
