#ifndef SO3_H
#define SO3_H

#include <Eigen/Dense>
#include <random>
#include <iostream>
#include <cmath>

#include "utils.h"

template <typename T>
class SO3
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Mat3T = Eigen::Matrix<T,3,3>;
    using Vec3T = Eigen::Matrix<T,3,1>;
    using Map3T = Eigen::Map<Mat3T>;
};

#endif
