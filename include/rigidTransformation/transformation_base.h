#ifndef TRANSFORMATION_BASE
#define TRANSFORMATION_BASE

#include <Eigen/Core>

//This class is just a base defining some functions that all inherited classes will contain
constexpr double PI = 3.14159265;

template <typename T> //, unsigned int S>
class TransformationBase
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
public:
    TransformationBase() = default;
    TransformationBase(const Eigen::MatrixBase<T> &mat): _arr{mat} {}
    virtual ~TransformationBase() = default;

    // Defined virtual methods
    virtual Eigen::MatrixBase<T> Mat() const { return _arr; }
    // virtual T* data() { return _arr.data(); }

protected:
    // Eigen::Matrix<T,S,S> _arr;
    Eigen::MatrixBase<T> _arr;
};

#endif