#ifndef SO3_H
#define SO3_H

#include <Eigen/Dense>
#include <random>
#include <iostream>
#include <cmath>

#include "utils.h"

template <typename T>
class SO3
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Mat3T = Eigen::Matrix<T,3,3>;
    using Vec3T = Eigen::Matrix<T,3,1>;
    using Map3T = Eigen::Map<Mat3T>;
public:
    SO3() = default;
    SO3(const Mat3T &mat): _arr{mat} {}
    SO3(const Eigen::AngleAxis<T> &mat): _arr{mat} {}
    SO3(const Eigen::Quaternion<T> &q): _arr{q} {}
    SO3(T* mat): _arr{Map3T(mat)} {}

    SO3 operator*(const SO3 &rhs) const
    {
        return SO3(this->R() * rhs.R());
    }

    SO3& operator*=(const SO3 &rhs)
    {
        (*this) = (*this) * rhs;
        return *this;
    }

    Vec3T operator*(const Vec3T &v) const
    {
        return (*this).R() * v;
    }

    bool operator==(const SO3 &rhs) const 
    {
        return this->R().isApprox(rhs.R());
    }

    Mat3T R() const { return _arr; }
    
    Mat3T Adj() const { return this->R(); }

    SO3 inv() const { return SO3(_arr.transpose()); }

    void selfInv() { _arr.transposeInPlace(); }

    bool isValidRotation() const 
    {
        T det{_arr.determinant()};
        return abs(det - 1.0) < 1e-8;
    }

    Vec3T rota(const Vec3T &v) const 
    {
        return (*this) * v;
    }

    Vec3T rotp(const Vec3T &v) const 
    {
        return this->inv() * v;
    }

    SO3 boxplus(const Vec3T &v) const 
    {
        return (*this) * SO3::Exp(v);
    }

    Vec3T boxminus(const SO3 &R) const 
    {
        return SO3::Log(R.inv() * (*this));
    }

    void normalize()
    {
        Vec3T x{this->R().col(0)}, y, z;
        x /= x.norm();
        y = x.cross(this->R().col(2));
        y /= y.norm();
        z = x.cross(y);

        _arr.col(0) = x;
        _arr.col(1) = y;
        _arr.col(2) = z;
    }

    T* data() { return _arr.data(); }
    const T* data() const { return _arr.data(); }

    static SO3 random() 
    {
        std::random_device rd;
        std::mt19937 generator(rd());
        std::uniform_real_distribution<T> dist(T(0), T(1));

        Vec3T x;
        x << dist(generator), dist(generator), dist(generator);

        T psi{ 2 * PI * x(0)};
        Mat3T R;
        R << cos(psi), sin(psi), T(0), -sin(psi), cos(psi), T(0), T(0), T(0), T(1);

        Vec3T v;
        v << cos(2 * PI * x(1)) * sqrt(x(2)), sin(2 * PI * x(1)) * sqrt(x(2)), sqrt(T(1) - x(2));

        Mat3T H{Mat3T::Identity() - 2 * v * v.transpose()};

        return SO3(-H * R);
    }

    static SO3 fromAxisAngle(const Vec3T &w)
    {
        T theta{w.norm()};
        Mat3T w_x{skew3(w)};

        T A,B;
        if(abs(theta) > 1e-6)
        {
            A = sin(theta) / theta;
            B = (1 - cos(theta)) / (pow(theta,2));
        }
        else // Taylor Series Expansion
        {
            A = 1.0 - pow(theta,2)/6.0 + pow(theta,4)/120.0;
            B = 0.5 - pow(theta,2)/24.0 + pow(theta,4)/720.0;
        }

        Mat3T R{Mat3T::Identity() + A * w_x + B * w_x * w_x};
        return SO3(R);
    }

    static SO3 fromAxisAngle(const Eigen::AngleAxis<T> &v)
    {
        return SO3(v);
    }

    static SO3 fromRPY(const Vec3T &v)
    {
        T phi{v(0)}, theta{v(1)}, psi{v(2)};

        T cp{cos(phi)}, sp{sin(phi)};
        T ct{cos(theta)}, st{sin(theta)};
        T cpsi{cos(psi)}, spsi{sin(psi)};

        Mat3T Rx, Ry, Rz;
        Rx << T(1.0), T(0.0), T(0.0), T(0.0), cp, -sp, T(0.0), sp, cp;
        Ry << ct, T(0.0), st, T(0.0), T(1.0), T(0.0), -st, T(0.0), ct;
        Rz << cpsi, -spsi, T(0.0), spsi, cpsi, T(0.0), T(0.0), T(0.0), T(1.0);

        return SO3(Rz * Ry * Rx);
    }

    static SO3 fromRPY(T& phi, T&theta, T&psi)
    {
        Vec3T angs;
        angs << phi, theta, psi;
        return SO3::fromRPY(angs);
    }

    static SO3 fromQuaternion(const Eigen::Matrix<T,4,1> &q) // q is a hamiltonian quaternion
    {
        T qw{q(0)};
        Vec3T qv{q.template tail<3>()};
        Mat3T qv_x{skew3(qv)};

        // Mat3T R{(2 * pow(qw,2) - 1.0) * Mat3T::Identity() - 2 * qw * qv_x + 2 * qv * qv.transpose()}; //Passive rotation. Need to check my implementation
        Mat3T R{(2 * pow(qw,2) - 1.0) * Mat3T::Identity() + 2 * qw * qv_x + 2 * qv * qv.transpose()}; //Active rotation
        return SO3(R);
    }

    static SO3 fromQuaternion(const Eigen::Quaternion<T> &q)
    {
        return SO3(q);
    }

    Mat3T log() const 
    {
        return SO3::log(this->R());
    }

    static Mat3T log(const SO3 &R)
    {
        return R.log();
    }

    static Mat3T log(const Mat3T &R)
    {
        T theta{acos((R.trace() -1.0)/2.0)};
        
        if(abs(theta) < 1e-6)
        {
            T temp{0.5 * (1 + pow(theta,2)/6.0 + 7 * pow(theta,4)/360.0)};
            return temp * (R - R.transpose());
        }
        else if(abs(abs(theta) - PI) < 1e-6) //Try 2 else if: 1 for less than pi the other for greater than pi
        {
            T th_m_PI{theta - PI};
            T temp{-PI/th_m_PI - 1.0 - PI/6.0 * th_m_PI - pow(th_m_PI,2)/6.0 - 7 * PI/360.0 * pow(th_m_PI,3) - 7/360.0 * pow(th_m_PI,4)};
            return temp/2.0 * (R - R.transpose()); //Not always great but hopefully close enough
            // Mat3T tempR{R.log()};
            // return R.log();
        }
        else
            return theta / (2.0 * sin(theta)) * (R - R.transpose());
    }

    Vec3T Log() const 
    {
        Mat3T log_R{this->log()};
        return SO3::vee(log_R);
    }

    static Vec3T Log(const SO3 &R)
    {
        return R.Log();
    }

    static Vec3T Log(const Mat3T &R)
    {
        Mat3T log_R{SO3::log(R)};
        return SO3::vee(log_R);
    }

    static SO3 exp(const Mat3T &log_R)
    {
        Vec3T w{SO3::vee(log_R)};
        T theta{w.norm()};

        Mat3T R;
        if(abs(theta) > 1e-6)
            R = Mat3T::Identity() + sin(theta)/theta * log_R + (1 - cos(theta))/pow(theta,2) * log_R * log_R;
        else 
        {
            T A{1.0 - pow(theta,2)/6.0 + pow(theta,4)/120.0};
            T B{0.5 - pow(theta,2)/24.0 + pow(theta,4)/720.0};
            R = Mat3T::Identity() + A * log_R + B * log_R * log_R;
        }
        return SO3(R);
    }

    static SO3 Exp(const Vec3T &w)
    {
        Mat3T log_R{SO3::hat(w)};
        return SO3::exp(log_R);
    }

    static Vec3T vee(const Mat3T &log_R)
    {
        Vec3T w;
        w << log_R(2,1), log_R(0,2), log_R(1,0);
        return w;
    }

    static Mat3T hat(const Vec3T &w)
    {
        return skew3(w);
    }

private:
    Mat3T _arr;
};

#endif