#ifndef SE2_H
#define SE2_H

#include <Eigen/Dense>
#include <random>
#include <iostream>
#include <cmath>

constexpr double PI = 3.141592653589793238462643383279;

template<typename F>
class SE2
{
    using Mat3F = Eigen::Matrix<F,3,3>;
    using RotF = Eigen::Matrix<F,2,2>;
    using transF = Eigen::Matrix<F,2,1>;
    using Vec3F = Eigen::Matrix<F,3,1>;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
public:
    SE2() = default;
    SE2(const Mat3F &mat): _arr{mat} {}
    SE2(const RotF &R, const transF &t)
    {
        _arr.template block<2,2>(0,0) = R;
        _arr.template block<2,1>(0,2) = t;
        _arr(2,2) = F(1.0);
    }

    Mat3F T() const { return _arr; }
    RotF R() const { return _arr.template block<2,2>(0,0); }
    transF t() const {return _arr.template block<2,1>(0,2); }

    bool isValidTransformation() const 
    {
        F det{this->R().determinant()};
        bool valid_rot{abs(det - F(1.0)) < 1e-8};
        bool homogeneous{this->T()(2,2) == F(1.0)};

        return (valid_rot && homogeneous);
    }

    static SE2 random()
    {
        std::random_device rd;
        std::mt19937 generator(rd());
        std::uniform_real_distribution<F> dist(F(-PI), F(PI)); //Could edit to get bigger range for translation

        F ang{dist(generator)};
        transF t;
        t << dist(generator), dist(generator);

        F ct{cos(ang)}, st{sin(ang)};
        RotF arr;
        arr << ct, -st, st, ct;
        return SE2(arr, t);
    }

    static SE2 fromAngleAndVec(const F ang, const transF &t)
    {
        RotF R;
        R << cos(ang), -sin(ang), sin(ang), cos(ang);
        return SE2(R, t);
    }

private:
    Mat3F _arr;
};

#endif