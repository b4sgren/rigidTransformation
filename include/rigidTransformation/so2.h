#ifndef SO2_H
#define SO2_H

#include "transformation_base.h"
// #include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <random>
#include <cmath>


template<typename T> //Replace all Matrix<T,2,2> with a using statement
class SO2 : public TransformationBase<T,2>
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using TransformationBase<T,2>::_arr;
    using Mat2T = Eigen::Matrix<T,2,2>;
public:
    SO2(): TransformationBase<T,2>{Eigen::Matrix<T,2,2>::Identity()} {}
    SO2(Eigen::Matrix<T, 2, 2> mat) : TransformationBase<T,2>{mat} {}

    SO2<T> operator*(SO2<T> R2)
    {
        return SO2<T>(this->R() * R2.R());
    }

    Eigen::Vector2d operator*(const Eigen::Vector2d &v)
    {
        return this->R() * v;
    }

    Eigen::Matrix<T,2,2> R() const { return _arr; }

    bool isValidRotation() const
    {
        double det = _arr.determinant();
        return abs(det - 1.0) < 1e-8;
    }

    SO2<T> inv()
    {
        return SO2<T>(_arr.transpose());
    }

    void selfInv()
    {
        _arr.transposeInPlace();
    }

    Eigen::Vector2d rota(const Eigen::Vector2d &v) //Template for use with Ceres?
    {
        return (*this) * v;
    }

    Eigen::Vector2d rotp(const Eigen::Vector2d &v) //Template for use with Ceres?
    {
        return this->inv() * v;
    }

    static SO2<T> random()
    {
        std::random_device rd;
        std::mt19937 generator(rd());
        std::uniform_real_distribution<T> dist(T(-PI), T(PI));

        T ang{dist(generator)};

        T ct{cos(ang)}, st{sin(ang)};
        Eigen::Matrix<T,2,2> arr;
        arr << ct, -st, st, ct;
        return SO2<T>(arr);
    }

    static SO2<T> fromAngle(T ang)
    {
        T ct{cos(ang)}, st{sin(ang)};
        Mat2T mat = (Mat2T() << ct, -st, st, ct).finished();
        return SO2<T>(mat);
    }

    static Eigen::Matrix2d hat(double ang) //Template for use with Ceres?
    {
        Eigen::Matrix2d mat;
        mat << 0.0, -ang, ang, 0.0;
        return mat;
    }

    static double vee(const Eigen::Matrix2d &mat) //Template for use with Ceres?
    {
        return mat(1,0);
    }

private:
};

#endif