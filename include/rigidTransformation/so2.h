#ifndef SO2_H
#define SO2_H

#include "transformation_base.h"
#include <Eigen/Core>
#include <iostream>


template<typename T> //Replace all Matrix<T,2,2> with a using statement
class SO2 : public TransformationBase<T,2>
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using TransformationBase<T,2>::_arr;
public:
    SO2(): TransformationBase<T,2>{Eigen::Matrix<T,2,2>::Identity()} {}
    SO2(Eigen::Matrix<T, 2, 2> mat) : TransformationBase<T,2>{mat} {}

    Eigen::Matrix<T,2,2> R() const { return _arr; }

private:
};

#endif