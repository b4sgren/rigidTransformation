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

    SO2(const Mat2T &R) : arr_(const_cast<T*>(R.data())) {}

    SO2(const T& ang): arr_(data_)
    {
        T ct{cos(ang)}, st{sin(ang)};
        arr_ << ct, -st, st, ct;
    }

    Mat2T R() const { return arr_; }

    T det() const
    {
        std::cout << arr_ << std::endl; // Get rid of this eventually
        return arr_.determinant();  //The memory is getting messed with. The two columns are the same
        // return data_[0] * data_[3] - data_[1] * data_[2];
    }

    static SO2 random()
    {
        // Stuff that uses this function is currently broken
        T ang{randomScalar(T(-PI), T(PI))};
        return SO2(ang);
    }

    Eigen::Map<Mat2T> arr_;
private:
    T data_[4];
};

template<typename T>
std::ostream& operator <<(std::ostream &os, const SO2<T> &R)
{
    os << R.R();
    return os;
}

}

#endif
