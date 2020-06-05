#ifndef TRANSFORMATION_BASE
#define TRANSFORMATION_BASE

#include <Eigen/Core>

//This class is just a base defining some functions that all inherited classes will contain
const double PI = 3.14159265;

template <typename T, unsigned int S>
class TransformationBase
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
public:
    TransformationBase() = default;
    TransformationBase(const Eigen::Matrix<T,S,S> &mat): _arr{mat} {}

    virtual Eigen::Matrix<T,S,S> R() const = 0;    

protected:
    Eigen::Matrix<T,S,S> _arr;
};

#endif