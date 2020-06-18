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

    // Defined virtual methods
    virtual Eigen::Matrix<T,S,S> Mat() const { return _arr; }
    virtual T* data() { return _arr.data(); }

    // Pure virtual methods
    virtual Eigen::Matrix<T,S,S> R() const = 0;    
    virtual Eigen::Matrix<T,S,S> Adj() const = 0; //This signature will have to change with SE<T>
    // virtual Eigen::Matrix<T,S,S> skew(const Eigen::Matrix<T,(S*(S-1))/2,1> &vec) const = 0; //Can this be virtual?

protected:
    Eigen::Matrix<T,S,S> _arr;
};

#endif