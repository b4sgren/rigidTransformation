#ifndef TRANSFORMATION_BASE
#define TRANSFORMATION_BASE

#include <Eigen/Core>

//This class is just a base defining some functions that all inherited classes will contain

template <class T>
class TransformationBase
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
public:
    TransformationBase(const Eigen::MatrixBase<T> &mat);
    virtual Eigen::MatrixBase<T> R() const;    

private:
    Eigen::MatrixBase<T> _arr;

};

#endif