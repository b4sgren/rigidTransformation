#ifndef TRANSFORMATION_BASE
#define TRANSFORMATION_BASE

#include <Eigen/Core>

//This class is just a base defining some functions that all inherited classes will contain

template <typename T>
class TransformationBase
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
public:
    TransformationBase();
    TransformationBase(T v): val{v} {}
    // virtual Eigen::MatrixBase<T> R() const;    
    virtual T R() const { return val; }

protected:
    // Eigen::MatrixBase<T> _arr;
    T val;
};

#endif