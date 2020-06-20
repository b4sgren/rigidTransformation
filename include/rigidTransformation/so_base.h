#ifndef SOBASE_H
#define SOBASE_H

// #include "transformation_base.h"
#include <Eigen/Core>
#include <cmath>

constexpr double PI = 3.14159265;

template <typename T, unsigned int S>
class SO_Base //: TransformationBase<T>
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using MatST = Eigen::Matrix<T,S,S>;
public:
    SO_Base() = default;
    SO_Base(const Eigen::Matrix<T,S,S> &mat): _arr{mat} {}
    virtual ~SO_Base() = default;

    MatST R() const { return _arr; }

    bool isValidRotation() const 
    {
        T det = _arr.determinant();
        return abs(det - 1.0) < 1e-8;
    }

    //virtual Functions

protected:
    MatST _arr;
};

#endif