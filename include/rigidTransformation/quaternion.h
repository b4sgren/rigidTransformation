#ifndef QUATERNION_H
#define QUATERNION_H

#include <Eigen/Core>
#include <random>
#include <iostream>
#include <cmath>

#include "utils.h"

namespace rigidTransform
{

template<typename T>
class Quaternion
{
public:
    using Vec4T = Eigen::Matrix<T,4,1>;
    using Mat3T = Eigen::Matrix<T,3,3>;
    using Vec3T = Eigen::Matrix<T,3,1>;

    Quaternion() : arr_(data_)
    {
        arr_(0) = T(1.0);
    }

    Vec4T q() const { return arr_; }

private:
    T data_[4];
public:
    Eigen::Map<Vec4T> arr_;
};
}

#endif
