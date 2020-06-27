#ifndef SE2_H
#define SE2_H

#include <Eigen/Dense>
#include <random>
#include <iostream>
#include <cmath>

constexpr double PI = 3.141592653589793238462643383279;

template<typename F>
class SE2
{
    using Mat3F = Eigen::Matrix<F,3,3>;
    using RotF = Eigen::Matrix<F,2,2>;
    using transF = Eigen::Matrix<F,2,1>;
    using Vec3F = Eigen::Matrix<F,3,1>;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
public:
    SE2() = default;
    SE2(const Mat3F &mat): _arr{mat} {}

    Mat3F T() const { return _arr; }
    RotF R() const { return _arr.template block<2,2>(0,0); }
    transF t() const {return _arr.template block<2,1>(0,2); }

private:
    Mat3F _arr;
};

#endif