#ifndef SOBASE_H
#define SOBASE_H

#include <Eigen/Core>
#include <cmath>

constexpr double PI = 3.14159265;

template <typename T, unsigned int S>
class SO_Base 
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

    SO_Base<T,S>& inv() //Can't return SO_Base<T,S> because abstract. Need this in this class for rotp...
    //Is inheritance the right setup. Should I just use templating?
    {
        return SO_Base<T, S>(_arr.transpose()); //This return type isn't valid b/c abstract
    }

    void selfInv()
    {
        _arr.transposeInPlace();
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
    virtual MatST log() const = 0;
    // virtual MatST Adj() const = 0;
    // virtual SO_Base<T,S> boxplus(const VecST &v) = 0;
    // virtual VecST boxminus(const SO_Base<T,S> R) = 0;

protected:
    MatST _arr;
};

#endif