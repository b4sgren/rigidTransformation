#ifndef SO3_H
#define SO3_H

#include <Eigen/Dense>
#include <random>
#include <iostream>

template <typename T>
class SO3
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Mat3T = Eigen::Matrix<T,3,3>;
    using Vec3T = Eigen::Matrix<T,3,1>;
public:
    SO3() = default;
    SO3(const Mat3T &mat): _arr{mat} {}

    Mat3T R() { return _arr; }

private:
    Mat3T _arr;
};

#endif