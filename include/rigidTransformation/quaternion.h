#ifndef QUATERNION_H
#define QUATERNION_H

#include <Eigen/Core>
#include <random>
#include <iostream>
#include <cmath>

#include "utils.h"

namespace rigidTransform
{

template<typename T>
class Quaternion
{
public:
    using Vec4T = Eigen::Matrix<T,4,1>;
    using Mat3T = Eigen::Matrix<T,3,3>;
    using Vec3T = Eigen::Matrix<T,3,1>;

    Quaternion() : arr_(data_)
    {
        arr_(0) = T(1.0);
    }

    Quaternion(const T* data) : arr_(const_cast<T*>(data)) {}

    Quaternion(const Vec4T &q) : arr_(data_) { arr_ = q; }

    Quaternion(const T& phi, const T& theta, const T& psi) : arr_(data_)
    {
        T cp{cos(phi/2.0)}, sp{sin(phi/2.0)};
        T ct{cos(theta/2.0)}, st{sin(theta/2.0)};
        T cps{cos(psi/2.0)}, sps{sin(psi/2.0)};

        Vec4T q{Vec4T::Zero()};
        q(0) = cps * ct * cp + sps * st * sp;
        q(1) = cps * ct * sp - sps * st * cp;
        q(2) = cps * st * cp + sps * ct * sp;
        q(3) = sps * ct * cp - cps * st * sp;

        if(q(0) < T(0.0))
            q *= -1;

        arr_ = q;
    }

    Quaternion(const Mat3T &R) : arr_(data_)
    {
        T d{R.trace()};
        Vec4T q;
        if(d > 0)
        {
            T s{2 * sqrt(d + 1)};
            q << s/4, (R(1,2) - R(2,1))/s, (R(2,0) - R(0,2))/s, (R(0,1) - R(1,0))/s;
        }
        else if(R(0,0) > R(1,1) && R(0,0) > R(2,2))
        {
            T s{2 * sqrt(1 + R(0,0) - R(1,1) - R(2,2))};
            q << (R(1,2) - R(2,1))/s , s/4, (R(1,0) + R(0,1))/s, (R(2,0) + R(0,2))/s;
        }
        else if(R(1,1) > R(2,2))
        {
            T s{2 * sqrt(1 + R(1,1) - R(0,0) - R(2,2))};
            q << (R(2,0) - R(0,2))/s, (R(1,0) + R(0,1))/s, s/4, (R(2,1) + R(1,2))/s;
        }
        else
        {
            T s{2 * sqrt(1 + R(2,2) - R(0,0) - R(1,1))};
            q << (R(0,1) - R(1,0))/s, (R(2,0) + R(0,2))/s, (R(2,1) + R(1,2))/s, s/4;
        }
        q.template tail<3>() *= -1;

        if(q(0) < T(0.0))
            q *= -1;

        arr_ = q;
    }

    Quaternion(const Vec3T &v) : arr_(data_)
    {
        T theta{v.norm()};
        Vec3T vec{v/theta};
        Vec4T q{Vec4T::Zero()};
        q(0) = cos(theta/2);
        q.template tail<3>() = vec * sin(theta/2);

        arr_ = q;
    }

    Quaternion(const Quaternion &q) : arr_(data_)
    {
        arr_ = q.q();
    }

    Quaternion& operator=(const Quaternion &rhs)
    {
        arr_ = rhs.q();
        return (*this);
    }

    Quaternion operator*(const Quaternion &rhs)
    {
        Eigen::Matrix<T,4,4> Q;
        Q(0,0) = qw();
        Q.template block<1,3>(0,1) = -qv().transpose();
        Q.template block<3,1>(1,0) = qv();
        Q.template block<3,3>(1,1) = qw() * Mat3T::Identity() + skew3<T>(qv());
        Vec4T qp = Q * rhs.q();

        if(qp(0) < T(0))
            qp *= -1;

        return Quaternion(qp);
    }

    Vec4T q() const { return arr_; }

    T qw() const { return arr_(0); }
    T qx() const { return arr_(1); }
    T qy() const { return arr_(2); }
    T qz() const { return arr_(3); }
    Vec3T qv() const { return arr_.template tail<3>(); }

    Mat3T R() const
    {
        Vec3T qv = this->qv();
        Mat3T R = (2 * pow(qw(),2) - 1) * Mat3T::Identity() - 2 * qw() * skew3<T>(qv) + 2 * qv * qv.transpose();
        return R;
    }

    T norm() const
    {
        return arr_.norm();
    }

    static Quaternion random()
    {
        Vec3T u;
        u << randomScalar<T>(0, 1), randomScalar<T>(0,1), randomScalar<T>(0,1);
        Vec4T q;
        q << sin(2 * PI * u(1)) * sqrt(1 - u(0)),
             cos(2 * PI * u(1)) * sqrt(1 - u(0)),
             sin(2 * PI * u(2)) * sqrt(u(0)),
             cos(2 * PI * u(2)) * sqrt(u(0));
        return Quaternion(q);
    }

    static Quaternion Identity()
    {
        return Quaternion();
    }

private:
    T data_[4];
public:
    Eigen::Map<Vec4T> arr_;
};

template<typename T>
std::ostream& operator <<(std::ostream &os, const Quaternion<T> &q)
{
    os << q.q().transpose();
    return os;
}
}

#endif
