#ifndef SO2_H
#define SO2_H

#include <Eigen/Dense>
#include <iostream>
#include <random>
#include <cmath>

constexpr double PI = 3.14159265;

template<typename T> 
class SO2
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Mat2T = Eigen::Matrix<T,2,2>;
    using Vec2T = Eigen::Matrix<T,2,1>;
public:
    SO2() = default;
    SO2(Eigen::Matrix<T, 2, 2> mat) : _arr{mat} {}

    SO2<T> operator*(SO2<T> R2) 
    {
        return SO2<T>(this->R() * R2.R());
    }

    Vec2T operator*(const Vec2T &v) const 
    {
        return this->R() * v;
    }

    bool operator==(const SO2<T> &rhs) const
    {
        return this->R().isApprox(rhs.R());
    }

    Mat2T R() const { return _arr; }

    Mat2T Adj() const { return Mat2T::Identity(); }

    bool isValidRotation() const 
    {
        T det{_arr.determinant()};
        return abs(det - 1.0) < 1e-8;
    }

    SO2<T> inv() { return SO2<T>(_arr.transpose()); }

    void selfInv() { _arr.transposeInPlace(); }

    Vec2T rota(const Vec2T &v)
    {
        return (*this) * v;
    }

    Vec2T rotp(const Vec2T &v)
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

    static Mat2T hat(T ang)
    {
        Mat2T mat;
        mat << T(0.0), -ang, ang, T(0.0);
        return mat;
    }

    static T vee(const Mat2T &mat)
    {
        return mat(1,0);
    }

    static Mat2T log(const Mat2T &R)
    {
        T theta{atan2(R(1,0), R(0,0))};
        Mat2T log_R;
        log_R << T(0.0), -theta, theta, T(0.0);
        return log_R;
    }

    static Mat2T log(const SO2<T> &R)
    {
        return R.log();
    }

    Mat2T log() const 
    {
        return SO2<T>::log(this->R());
    }

    static T Log(const Mat2T &R)
    {
        Mat2T log_R{SO2<T>::log(R)};
        return SO2<T>::vee(log_R);
    }

    static T Log(const SO2<T> &R)
    {
        Mat2T log_R{SO2<T>::log(R)};
        return SO2<T>::vee(log_R);
    }

    T Log() const 
    {
        Mat2T log_R{this->log()};
        return SO2<T>::vee(log_R);
    }

    static SO2<T> exp(const Mat2T &R)
    {
        T theta{SO2<T>::vee(R)};
        return SO2<T>::fromAngle(theta);
    }

    static SO2<T> Exp(const double phi)
    {
        return SO2<T>::fromAngle(phi);
    }

private:
    Mat2T _arr;
};

#endif