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

    SO3(T phi, T theta, T psi) : arr_(data_)
    {
        T cphi{cos(phi)}, sphi(sin(phi));
        T ct{cos(theta)}, st{sin(theta)};
        T cpsi{cos(psi)}, spsi{sin(psi)};

        Mat3T Rpsi, Rtheta, Rphi;
        Rpsi << cpsi, -spsi, 0, spsi, cpsi, 0, 0, 0, 1;
        Rtheta << ct, 0, st, 0, 1, 0, -st, 0, ct;
        Rphi << 1, 0, 0 , 0, cphi, -sphi, 0, sphi, cphi;

        arr_ = Rpsi * Rtheta * Rphi;
    }

    Mat3T R() const { return arr_; }

private:
    T data_[9];
public:
    Eigen::Map<Mat3T> arr_;
};

}

#endif
