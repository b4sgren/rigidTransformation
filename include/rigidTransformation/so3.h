#ifndef SO3_H
#define SO3_H

#include <Eigen/Dense>
#include <random>
#include <iostream>
#include <cmath>

#include "utils.h"

namespace rigidTransform
{

template <typename T>
class SO3
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Mat3T = Eigen::Matrix<T,3,3>;
    using Vec3T = Eigen::Matrix<T,3,1>;
    using Map3T = Eigen::Map<Mat3T>;

    SO3(): arr_(data_)
    {
        arr_.setIdentity();
    }

    SO3(const T* data) : arr_(const_cast<T*>(data)) {}

    SO3(const Mat3T &R) : arr_(data_) { arr_ = R; }

    SO3(const SO3 &R) : arr_(data_)
    {
        arr_ = R.R();
    }

    SO3(T phi, T theta, T psi) : arr_(data_)
    {
        T cphi{cos(phi)}, sphi(sin(phi));
        T ct{cos(theta)}, st{sin(theta)};
        T cpsi{cos(psi)}, spsi{sin(psi)};

        Mat3T Rpsi, Rtheta, Rphi;
        Rpsi << cpsi, -spsi, 0, spsi, cpsi, 0, 0, 0, 1;
        Rtheta << ct, 0, st, 0, 1, 0, -st, 0, ct;
        Rphi << 1, 0, 0 , 0, cphi, -sphi, 0, sphi, cphi;

        arr_ = Rpsi * Rtheta * Rphi;
    }

    SO3(const Vec3T &v) : arr_{data_}
    {
        T theta = v.norm();
        Mat3T vx{skew3<T>(v)};

        T A{1.0}, B{0.5};
        if(abs(theta) > 1e-8)
        {
            A = sin(theta) / theta;
            B = (1 - cos(theta)) / (theta * theta);
        }

        arr_ = Mat3T::Identity() + A * vx + B * vx * vx;
    }

    Mat3T R() const { return arr_; }

    Mat3T Adj() const { return arr_; }

    SO3& operator=(const SO3& rhs)
    {
        arr_ = rhs.R();
        return (*this);
    }

    SO3 operator*(const SO3 &rhs)
    {
        SO3 temp;
        temp.arr_ = R() * rhs.R();
        return temp;
    }

    SO3& operator*=(const SO3 &rhs)
    {
        (*this) = (*this) * rhs;
        return (*this);
    }

    SO3 inverse() const
    {
        SO3 temp;
        temp.arr_ = R().transpose();
        return temp;
    }

    void inverse_()
    {
        arr_.transposeInPlace();
    }

    Vec3T rota(const Eigen::Ref<const Vec3T>& v)
    {
        return R() * v;
    }

    Vec3T rotp(const Eigen::Ref<const Vec3T>& v)
    {
        return inverse().R() * v;
    }

    SO3 boxplusr(const Eigen::Ref<const Vec3T> &v)
    {
        return (*this) * SO3::Exp(v);
    }

    Vec3T boxminusr(const SO3 &R2)
    {
        return SO3::Log(R2.inverse() * (*this));
    }

    static SO3 random()
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dist(0,1);

        Vec3T x;
        x << dist(gen), dist(gen), dist(gen);
        T psi{2 * PI * x(0)};
        Mat3T Rpsi;
        Rpsi << cos(psi), sin(psi), 0, -sin(psi), cos(psi), 0, 0, 0, 1;

        Vec3T v;
        v << cos(2*PI*x(1)) * sqrt(x(2)), sin(2*PI*x(1)) * sqrt(x(2)), sqrt(1 - x(2));
        Mat3T H = Mat3T::Identity() - 2 * v * v.transpose();
        Mat3T res = -H * Rpsi;
        return SO3(res);
    }

    static SO3 Identity()
    {
        return SO3();
    }

    Vec3T Log() const
    {
        return SO3::Log(*this);
    }

    static Vec3T Log(const SO3& R)
    {
        T theta = acos((R.R().trace() - 1)/2.0);
        Mat3T logR{Mat3T::Zero()};
        if(abs(theta) < 1e-8)
        {
            logR = 0.5 * (R.R() - R.R().transpose());
        }
        else if(abs(theta - PI) < 1e-8)
        {
            // Find a way to simplify this
        }
        else
        {
            logR = theta / (2 * sin(theta)) * (R.R() - R.R().transpose());
        }

        return SO3::vee(logR);
    }

    static SO3 Exp(const Eigen::Ref<const Vec3T> &logR)
    {
        Mat3T logRx{skew3<T>(logR)};
        T theta{logR.norm()};
        Mat3T R{Mat3T::Identity()};
        if(abs(theta) > 1e-8)
        {
            R = Mat3T::Identity() + sin(theta)/theta * logRx + (1 - cos(theta))/pow(theta,2) * logRx * logRx;
        }
        return SO3(R);
    }

    static Vec3T vee(const Eigen::Ref<const Mat3T>& M)
    {
        Vec3T v;
        v << M(2,1), M(0,2), M(1,0);
        return v;
    }

private:
    T data_[9];
public:
    Eigen::Map<Mat3T> arr_;
};

template<typename T>
std::ostream& operator <<(std::ostream &os, const SO3<T> &R)
{
    os << R.R();
    return os;
}
}

#endif
