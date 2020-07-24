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

    bool operator==(const Quaternion &rhs)
    {
        return this->q().isApprox(rhs.q());
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

    static Quaternion fromAxisAngle(const Eigen::AngleAxis<T> &mat)
    {
        Mat3T mat_{mat};
        return Quaternion::fromRotationMatrix(mat_);
    }

    static Quaternion fromAxisAngle(const Vec3T &vec)
    {
        T ang{vec.norm()};
        Vec3T v{vec / ang};

        QuatT q;
        if(abs(ang) > 1e-6)
            q << cos(ang/2.0), v * sin(ang/2.0);
        else 
        {
            T qw{1.0 - pow(ang,2)/8.0 + pow(ang,4)/46080.0};
            T temp{0.5 - pow(ang,2)/48.0 + pow(ang,4)/3840};
            Vec3T qv{vec * temp};
            q << qw, qv;
        }
        return Quaternion(q);
    }

    static Quaternion fromRotationMatrix(const Mat3T &R)
    {
        T d{R.trace()}, s;
        QuatT q;
        
        if(d > 0)
        {
            s = 2.0 * sqrt(d + 1.0);
            q << s/4.0, 1/s * (R(1,2) - R(2,1)), 1/s * (R(2,0) - R(0,2)), 1/s * (R(0,1) - R(1,0));
        }
        else if(R(0,0) > R(1,1) && R(0,0) > R(2,2))
        {
            s = 2 * sqrt(1 + R(0,0) - R(1,1) - R(2,2));
            q << 1/s * (R(1,2) - R(2,1)), s/4, 1/s * (R(1,0) + R(0,1)), 1/s * (R(2,0) + R(0,2));
        }
        else if(R(1,1) > R(2,2))
        {
            s = 2 * sqrt(1 + R(1,1) - R(0,0) - R(2,2));
            q << 1/s * (R(2,0) - R(0,2)), 1/s * (R(1,0) + R(0,1)), s/4, 1/s * (R(2,1) + R(1,2));
        }
        else 
        {
            s = 2 * sqrt(1 + R(2,2) - R(0,0) - R(1,1));
            q << 1/s * (R(0,1) - R(1,0)), 1/s * (R(2,0) + R(0,2)), 1/s * (R(2,1) + R(1,2)), s/4;
        }
        q.template segment<3>(1) *= -1;

        return Quaternion(q);
    }

    static Quaternion fromRPY(T &phi, T &theta, T &psi)
    {
        Vec3T rpy;
        rpy << phi, theta, psi;
        return Quaternion::fromRPY(rpy);
    }

    static Quaternion fromRPY(const Vec3T &rpy)
    {
        T phi{rpy(0)/2.0}, theta{rpy(1)/2.0}, psi{rpy(2)/2.0};

        T cp{cos(phi)}, sp{sin(phi)};
        T ct{cos(theta)}, st{sin(theta)};
        T cpsi{cos(psi)}, spsi{sin(psi)};

        QuatT q;
        q << cpsi * ct * cp + spsi * st * sp, 
             cpsi * ct * sp - spsi * st * cp, 
             cpsi * st * cp + spsi * ct * sp,
             spsi * ct * cp - cpsi * st * sp;
        
        return Quaternion(q);
    }

    static Vec3T vee(const QuatT &q)
    {
        return q.template segment<3>(1);
    }

    static QuatT hat(const Vec3T &v)
    {
        QuatT q;
        q << T(0.0), v;
        return q;
    }

    static Quaternion exp(const QuatT &log_q) //Possibly think about changing this to just accept the 3 vector...
    {
        Vec3T v{Quaternion::vee(log_q)};
        return Quaternion::fromAxisAngle(v);
    }

    static Quaternion Exp(const Vec3T &v)
    {
        return Quaternion::exp(Quaternion::hat(v));
    }

    static QuatT log(const Quaternion &q)
    {
        return q.log();
    }

    QuatT log() const 
    {
        T qw{this->qw()};
        Vec3T qv{this->qv()};

        T theta{qv.norm()};
        Vec3T w;
        if(abs(theta) > 1e-6)
        {
            w = 2 * atan(theta/qw) * qv/theta;
        }
        else 
        {
            T temp{1.0/qw - pow(theta,2) / (3 * pow(qw,3)) + pow(theta,4)/(5 * pow(qw,5))};
            w = 2 * temp * qv;
        }

        QuatT log_q;
        log_q << T(0.0), w;
        return log_q;
    }

    static Vec3T Log(const Quaternion &q)
    {
        return q.Log();
    }

    Vec3T Log() const 
    {
        QuatT log_q{this->log()};
        return Quaternion::vee(log_q);
    }

private:
    QuatT _arr;
};

#endif
