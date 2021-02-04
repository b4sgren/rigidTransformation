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
    SO3(const Mat3T &R) : arr_(const_cast<T*>(R.data())) {}

    SO3(const SO3 &R) : arr_(data_)
    {
        arr_ = R.R();
    }

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

    SO3(const Vec3T &v) : arr_{data_}
    {
        T theta = v.norm();
        Mat3T vx{skew3<T>(v)};

        T A{1.0}, B{0.5};
        if(abs(theta) > 1e-8)
        {
            A = sin(theta) / theta;
            B = (1 - cos(theta)) / (theta * theta);
        }

        arr_ = Mat3T::Identity() + A * vx + B * vx * vx;
    }

    Mat3T R() const { return arr_; }

    SO3& operator=(const SO3& rhs)
    {
        arr_ = rhs.R();
        return (*this);
    }

    SO3 operator*(const SO3 &rhs)
    {
        SO3 temp;
        temp.arr_ = R() * rhs.R();
        return temp;
    }

    SO3& operator*=(const SO3 &rhs)
    {
        (*this) = (*this) * rhs;
        return (*this);
    }

    SO3 inverse() const
    {
        SO3 temp;
        temp.arr_ = R().transpose();
        return temp;
    }

    static SO3 random()
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dist(0,1);

        Vec3T x;
        x << dist(gen), dist(gen), dist(gen);
        T psi{2 * PI * x(0)};
        Mat3T Rpsi;
        Rpsi << cos(psi), sin(psi), 0, -sin(psi), cos(psi), 0, 0, 0, 1;

        Vec3T v;
        v << cos(2*PI*x(1)) * sqrt(x(2)), sin(2*PI*x(1)) * sqrt(x(2)), sqrt(1 - x(2));
        Mat3T H = Mat3T::Identity() - 2 * v * v.transpose();
        Mat3T res = -H * Rpsi;
        return SO3(res);
    }

    static SO3 Identity()
    {
        return SO3();
    }

private:
    T data_[9];
public:
    Eigen::Map<Mat3T> arr_;
};

}

#endif
