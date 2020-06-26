#ifndef SO3_H
#define SO3_H

#include <Eigen/Dense>
#include <random>
#include <iostream>
#include <cmath>

constexpr double PI = 3.14159265;

template <typename T>
class SO3
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Mat3T = Eigen::Matrix<T,3,3>;
    using Vec3T = Eigen::Matrix<T,3,1>;
public:
    SO3() = default;
    SO3(const Mat3T &mat): _arr{mat} {}

    Mat3T R() { return _arr; }

    bool isValidRotation() const 
    {
        T det{_arr.determinant()};
        return abs(det - 1.0) < 1e-8;
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

        return SO3<T>(-H * R);
    }

private:
    Mat3T _arr;
};

#endif