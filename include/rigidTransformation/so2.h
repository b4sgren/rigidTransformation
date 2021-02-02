#ifndef SO2_H
#define SO2_H
#include <Eigen/Dense>
#include <iostream>
#include <random>
#include <cmath>

#include "utils.h"

namespace rigidTransform
{

template<typename T>
class SO2
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Mat2T = Eigen::Matrix<T,2,2>;
    using Vec2T = Eigen::Matrix<T,2,1>;
    using Map2T = Eigen::Map<Mat2T>;

    SO2() : arr_(data_)
    {
        arr_.setIdentity();
    }

    SO2(const T *data) : arr_(const_cast<T*>(data)) {}

    // I might need to be careful that what is passed in doesn't go out of scope and get destroyed. I might want to wrap arr_ around data_ like in the SO2 constructor
    SO2(const Mat2T &R) : arr_(const_cast<T*>(R.data())) {}

    SO2(const T& ang): arr_(data_)
    {
        T ct{cos(ang)}, st{sin(ang)};
        arr_ << ct, -st, st, ct;
    }

    SO2(const SO2 &R): arr_(data_)
    {
        arr_ = R.R();
    }

    SO2& operator=(const SO2 &rhs)
    {
        arr_ = rhs.R();
        return *this;
    }

    SO2 operator*(const SO2 &rhs) const
    {
        SO2 temp;
        temp.arr_ = R() * rhs.R();
        return temp;
    }

    SO2& operator*=(const SO2 &rhs)
    {
        (*this) = (*this) * rhs;
        return (*this);
    }

    Mat2T R() const { return arr_; }

    T det() const
    {
        return arr_.determinant();
    }

    SO2 inverse() const
    {
        SO2 temp;
        temp.arr_ = R().transpose();
        return temp;
    }

    // operations with _ at the end are in place
    void inverse_()
    {
        arr_.transposeInPlace();
    }

    Vec2T rota(const Vec2T& v) const
    {
        return R() * v;
    }

    Vec2T rotp(const Vec2T &v) const
    {
        return inverse().R() * v;
    }

    static SO2 random()
    {
        T ang{randomScalar(T(-PI), T(PI))};
        return SO2(ang);
    }

    static SO2 identity()
    {
        return SO2();
    }

private:
    T data_[4];
public:
    Eigen::Map<Mat2T> arr_;
};

template<typename T>
std::ostream& operator <<(std::ostream &os, const SO2<T> &R)
{
    os << R.R();
    return os;
}

}

#endif
