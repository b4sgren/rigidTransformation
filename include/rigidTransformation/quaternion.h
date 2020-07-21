#ifndef QUATERNION_H
#define QUATERNION_H

#include <Eigen/Core>
#include <random>
#include <iostream>
#include <cmath>

#include "utils.h"

template<typename T>
class Quaternion
{
    using QuatT = Eigen::Matrix<T,4,1>;
    using Mat3T = Eigen::Matrix<T,3,3>;
    using Vec3T = Eigen::Matrix<T,3,1>;
public:
    Quaternion() = default;
    Quaternion(const QuatT &q): _arr{q} { rectifyQuat(); } 

    Quaternion operator*(const Quaternion &rhs)
    {
        return this->otimes(rhs);
    }

    inline T qw() const { return _arr(0); }
    inline T qx() const { return _arr(1); }
    inline T qy() const { return _arr(2); }
    inline T qz() const { return _arr(3); }
    inline Vec3T qv() const { return _arr.template segment<3>(1); }
    inline QuatT q() const { return _arr; }

    bool isValidQuaternion() const 
    {
        T norm{_arr.norm()};
        return abs(1.0 - norm) < 1e-8 && _arr(0) >= 0.0;
    }

    void rectifyQuat()
    {
        if(_arr(0) < 0.0)
            _arr *= -1.0;
    }

    Mat3T R() const 
    {
        Mat3T I{Mat3T::Identity()};
        Vec3T q_v{qv()};
        return (2 * pow(qw(),2) - 1.0) * I + 2 * qw() * skew3(q_v) + 2 * q_v * q_v.transpose();
    }

    Quaternion otimes(const Quaternion &rhs)
    {
        Vec3T q_v{qv()};
        Mat3T I{Mat3T::Identity()};

        Eigen::Matrix<T,4,4> Q{Eigen::Matrix<T,4,4>::Zero()};
        Q(0,0) = qw();
        Q.template block<1,3>(0,1) = -q_v.transpose();
        Q.template block<3,1>(1,0) = q_v;
        Q.template block<3,3>(1,1) = qw() * I + skew3(q_v);
        return Quaternion(Q * rhs.q());
    }

    Quaternion inv() const 
    {
        QuatT q_inv;
        q_inv << qw(), -qv();

        return Quaternion(q_inv);
    }

    void selfInv()
    {
        _arr.template segment<3>(1) *= -1;
    }

    Vec3T rota(const Vec3T &v) const
    {
        T qw{this->qw()};
        Vec3T qv{this->qv()};

        Vec3T t{2 * skew3(v) * qv};
        return v - qw * t + skew3(t) * qv;
    }

    Vec3T rotp(const Vec3T &v) const
    {
        return this->inv().rota(v);
    }

    static Quaternion random()
    {
        std::random_device rd;
        std::mt19937 generator(rd());
        std::uniform_real_distribution<T> dist(0.0, 1.0);

        Vec3T u;
        u << dist(generator), dist(generator), dist(generator);

        T val1{sqrt(1.0 - u(0))}, val2{sqrt(u(0))};
        T qw{sin(2 * PI * u(1)) * val1};
        T qx{cos(2 * PI * u(1)) * val1};
        T qy{sin(2 * PI * u(2)) * val2};
        T qz{cos(2 * PI * u(2)) * val2};
        QuatT q;
        q << qw, qx, qy, qz;

        return Quaternion(q);
    }

private:
    QuatT _arr;
};

#endif
