#ifndef SOBASE_H
#define SOBASE_H

// #include "transformation_base.h"
#include <Eigen/Core>

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

protected:
    MatST _arr;
};

#endif