#ifndef SO2_H
#define SO2_H

#include "so_base.h"
#include <Eigen/Dense>
#include <iostream>
#include <random>
#include <cmath>


template<typename T> 
class SO2 : public SO_Base<T,2>
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Mat2T = Eigen::Matrix<T,2,2>;
    using Vec2T = Eigen::Matrix<T,2,1>;
public:
    SO2(): SO_Base<T,2>{Eigen::Matrix<T,2,2>::Identity()} {}
    SO2(Eigen::Matrix<T, 2, 2> mat) : SO_Base<T,2>{mat} {}
    SO2(const SO_Base<T,2> &R) : SO_Base<T,2>(R) {}

    SO2<T> operator*(SO2<T> R2) //Can this be inherited somehow??
    {
        return SO2<T>(this->R() * R2.R());
    }

    // Vec2T operator*(const Vec2T &v)
    // {
    //     return this->R() * v;
    // }

    // Mat2T Adj() const override { return Mat2T::Identity(); }

    // bool isValidRotation() const
    // {
    //     double det = _arr.determinant(); //double or type T?
    //     return abs(det - 1.0) < 1e-8;
    // }

    // SO2<T> inv()
    // {
    //     return SO2<T>(_arr.transpose());
    // }

    // void selfInv()
    // {
    //     _arr.transposeInPlace();
    // }

    // Vec2T rota(const Vec2T &v) //Should this be inherited?
    // {
    //     return (*this) * v;
    // }

    // Vec2T rotp(const Vec2T &v) //should this be inherited?
    // {
    //     return this->inv() * v;
    // }

    static SO2<T> random()
    {
        std::random_device rd;
        std::mt19937 generator(rd());
        std::uniform_real_distribution<T> dist(T(-PI), T(PI));

        T ang{dist(generator)};

        T ct{cos(ang)}, st{sin(ang)};
        Mat2T arr;
        arr << ct, -st, st, ct;
        return SO2<T>(arr);
    }

    static SO2<T> fromAngle(T ang)
    {
        T ct{cos(ang)}, st{sin(ang)};
        Mat2T mat = (Mat2T() << ct, -st, st, ct).finished();
        return SO2<T>(mat);
    }

    // static Mat2T hat(T ang)
    // {
    //     Mat2T mat;
    //     mat << T(0.0), -ang, ang, T(0.0);
    //     return mat;
    // }

    // static T vee(const Mat2T &mat)
    // {
    //     return mat(1,0);
    // }

private:
};

#endif