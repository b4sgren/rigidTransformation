#ifndef SO2_H
#define SO2_H

#include "transformation_base.h"
// #include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <random>
#include <cmath>


template<typename T> //Replace all Matrix<T,2,2> with a using statement
class SO2 : public TransformationBase<T,2>
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using TransformationBase<T,2>::_arr;
    // using Mat2T = Eigen::Matrix<T,2,2>;
public:
    SO2(): TransformationBase<T,2>{Eigen::Matrix<T,2,2>::Identity()} {}
    SO2(Eigen::Matrix<T, 2, 2> mat) : TransformationBase<T,2>{mat} {}

    Eigen::Matrix<T,2,2> R() const { return _arr; }

    static SO2<T> random()
    {
        std::random_device rd;
        std::mt19937 generator(rd());
        std::uniform_real_distribution<T> dist(T(-PI), T(PI));

        T ang{dist(generator)};

        T ct{cos(ang)}, st{sin(ang)};
        Eigen::Matrix<T,2,2> arr;
        arr << ct, -st, st, ct;
        return SO2<T>(arr);
    }

    bool isValidRotation() const
    {
        double det = _arr.determinant();
        return abs(det - 1.0) < 1e-8;
    }

private:
};

#endif