#ifndef SE3_H
#define SE3_H

#include <random>
#include <iostream>
#include <cmath>

#include <eigen3/Eigen/Dense>

#include "utils.h"

template<typename F>
class SE3
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Mat4F = Eigen::Matrix<F,4,4>;
    using Mat3F = Eigen::Matrix<F,3,3>;
    using Vec3F = Eigen::Matrix<F,3,1>;
    using Vec6F = Eigen::Matrix<F,6,1>;
public:
    SE3(): _arr{Mat4F::Identity()} {}
    SE3(const Mat4F &mat): _arr{mat} {}

    Mat4F T() { return _arr; }

private:
    Mat4F _arr;
};

#endif