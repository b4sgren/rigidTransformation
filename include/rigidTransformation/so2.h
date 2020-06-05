#ifndef SO2_H
#define SO2_H

#include "transformation_base.h"
#include <Eigen/Core>
#include <iostream>


template<typename T> //Replace all Matrix<T,2,2> with a using statement
class SO2 : public TransformationBase<T>
{
    using TransformationBase<T>::val;
public:
    // SO2(Eigen::Matrix<T, 2, 2> mat) : _arr{mat} {}
    // Eigen::Matrix<T,2,2> R() const { return _arr; }
    SO2(T v): TransformationBase<T>{v} {}
    // SO2(T v): val{v} {}
    // SO2(T v) { val = v; }
    T R() const  { return val; }

private:
};

#endif