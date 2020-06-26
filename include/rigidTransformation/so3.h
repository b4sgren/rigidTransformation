#ifndef SO3_H
#define SO3_H

#include <Eigen/Dense>
#include <random>
#include <iostream>
#include <cmath>

constexpr double PI = 3.14159265;

template <typename T>
Eigen::Matrix<T,3,3> skew(const Eigen::Matrix<T,3,1> &v)
{
    Eigen::Matrix<T,3,3> v_x;
    v_x << T(0), -v(2), v(1), v(2), T(0), -v(0), -v(1), v(0), T(0);
    return v_x;
}

template <typename T>
class SO3
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Mat3T = Eigen::Matrix<T,3,3>;
    using Vec3T = Eigen::Matrix<T,3,1>;
public:
    SO3() = default;
    SO3(const Mat3T &mat): _arr{mat} {}
    SO3(const Eigen::AngleAxis<T> &mat): _arr{mat} {}
    SO3(const Eigen::Quaternion<T> &q): _arr{q} {}

    SO3 operator*(const SO3 &rhs) const
    {
        return SO3(this->R() * rhs.R());
    }

    Vec3T operator*(const Vec3T &v) const
    {
        return (*this).R() * v;
    }

    Mat3T R() const { return _arr; }

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
        Mat3T w_x{skew(w)};

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
        Mat3T qv_x{skew(qv)};

        // Mat3T R{(2 * pow(qw,2) - 1.0) * Mat3T::Identity() - 2 * qw * qv_x + 2 * qv * qv.transpose()}; //Passive rotation. Need to check my implementation
        Mat3T R{(2 * pow(qw,2) - 1.0) * Mat3T::Identity() + 2 * qw * qv_x + 2 * qv * qv.transpose()}; //Active rotation
        return SO3(R);
    }

    static SO3 fromQuaternion(const Eigen::Quaternion<T> &q)
    {
        return SO3(q);
    }

private:
    Mat3T _arr;
};

#endif