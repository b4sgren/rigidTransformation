#ifndef SO3_H
#define SO3_H

#include <Eigen/Dense>
#include <random>
#include <iostream>
#include <cmath>

#include "utils.h"

namespace rigidTransform
{

template <typename T>
class SO3
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Mat3T = Eigen::Matrix<T,3,3>;
    using Vec3T = Eigen::Matrix<T,3,1>;
    using Map3T = Eigen::Map<Mat3T>;

    SO3(): arr_(data_)
    {
        arr_.setIdentity();
    }

    SO3(const T* data) : arr_(const_cast<T*>(data)) {}

    // This may lead to memory errors if the matrix R goes out of scope. May want to wrap arr_ around data. See so2
    SO3(const Mat3T & R) : arr_(const_cast<T*>(R.data())) {}

    Mat3T R() const { return arr_; }

private:
    T data_[9];
public:
    Eigen::Map<Mat3T> arr_;
};

}

#endif
