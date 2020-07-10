#ifndef QUATERNION_H
#define QUATERNION_H

#include <Eigen/Core>
#include <random>
#include <iostream>
#include <cmath>

template<typename T>
class Quaternion
{
    using QuatT = Eigen::Matrix<T,4,1>;
    using Mat3T = Eigen::Matrix<T,3,3>;
    using Vec3T = Eigen::Matrix<T,3,1>;
public:
    Quaternion() = default;
    Quaternion(const QuatT &q): _arr{q} {}

    QuatT q() const { return _arr; }

private:
    QuatT _arr;
};

#endif
