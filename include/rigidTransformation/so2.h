#ifndef SO2_H
#define SO2_H

#include "transformation_base.h"
#include <Eigen/Core>

template<class T> //Replace all Matrix<T,2,2> with a using statement
class SO2 : public TransformationBase<T>
{
public:
    SO2(Eigen::Matrix<T, 2, 2> mat) : _arr{mat} {}

    Eigen::Matrix<T,2,2> R() const { return _arr; }

private:
};

#endif