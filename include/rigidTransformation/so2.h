#ifndef SO2_H
#define SO2_H

#include <Eigen/Dense>
#include <iostream> // TODO Add << operator to print
#include <random>
#include <cmath>

#include "utils.h"

namespace rigidTransform
{

template<typename T>
class SO2
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Mat2T = Eigen::Matrix<T,2,2>;
    using Vec2T = Eigen::Matrix<T,2,1>;
    using Map2T = Eigen::Map<Mat2T>;
public:
    SO2() : arr_(data_)
    {
        arr_.setIdentity();
    }

    SO2(const T *data) : arr_(const_cast<T*>(data)) {}

    Mat2T R() const { return arr_; }

private:
    T data_[4];
    Eigen::Map<Mat2T> arr_;
};

}

#endif
