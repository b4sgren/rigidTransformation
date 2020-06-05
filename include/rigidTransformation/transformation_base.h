#ifndef TRANSFORMATION_BASE
#define TRANSFORMATION_BASE

#include <Eigen/Core>

//This class is just a base defining some functions that all inherited classes will contain

template <typename T>
class TransformationBase
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
public:
    TransformationBase() = default;
    TransformationBase(T v): val{v} {}
    // virtual Eigen::MatrixBase<T> R() const;    
    virtual T R() const = 0; //{ return val; } //=0; makes it a pure virtual function meaning that it has to be overridden in the base class

protected:
    // Eigen::MatrixBase<T> _arr;
    T val;
};

#endif