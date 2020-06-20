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
    using VecST = Eigen::Matrix<T,S,1>;
public:
    SO_Base() = default;
    SO_Base(const Eigen::Matrix<T,S,S> &mat): _arr{mat} {}
    virtual ~SO_Base() = default;

    MatST R() const { return _arr; }

    VecST operator*(const VecST &v)
    {
        return this->R() * v;
    }

    bool isValidRotation() const 
    {
        T det = _arr.determinant();
        return abs(det - 1.0) < 1e-8;
    }

    SO_Base<T,S> inv()
    {
        return SO_Base<T, S>(_arr.transpose());
    }

    VecST rota(const VecST &v)
    {
        return (*this) * v;
    }

    VecST rotp(const VecST &v)
    {
        return this->inv() * v;
    }

    //virtual Functions

protected:
    MatST _arr;
};

#endif