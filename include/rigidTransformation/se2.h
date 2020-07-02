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
    //Add a constructor that takes in an Eigen::Affine something and taking a F*
    SE2() = default;
    SE2(const Mat3F &mat): _arr{mat} {}
    SE2(const RotF &R, const transF &t)
    {
        _arr.template block<2,2>(0,0) = R;
        _arr.template block<2,1>(0,2) = t;
        _arr(2,2) = F(1.0);
    }

    SE2 operator*(const SE2 &rhs)
    {
        return SE2(this->T() * rhs.T());
    }

    Vec3F operator*(const Vec3F &v)
    {
        return this->T() * v;
    }

    transF operator*(const transF &v)
    {
        Vec3F v1;
        v1 << v(0), v(1), F(1.0);
        Vec3F vp{(*this) * v1};
        return vp.template segment<2>(0);
    }

    Mat3F T() const { return _arr; }
    RotF R() const { return _arr.template block<2,2>(0,0); }
    transF t() const {return _arr.template block<2,1>(0,2); }

    SE2 inv() const
    {
        RotF R_inv{this->R().transpose()};
        transF t_inv{-R_inv * this->t()};
        return SE2(R_inv, t_inv);
    }

    void selfInv()
    {
        _arr.template block<2,2>(0,0) = this->R().transpose();
        _arr.template block<2,1>(0,2) = -this->R() * this->t();
    }

    Vec3F transa(const Vec3F &v)
    {
        return (*this) * v;
    }

    Vec3F transp(const Vec3F &v)
    {
        return (*this).inv() * v;
    }

    transF transa(const transF &v)
    {
        return (*this) * v;
    }

    transF transp(const transF &v)
    {
        return (*this).inv() * v;
    }

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

    static RotF skew(const F &val)
    {
        RotF val_x;
        val_x << F(0.0), -val, val, F(0.0);
        return val_x;
    }

    static Mat3F hat(const Vec3F &w)
    {
        Mat3F log_T{Mat3F::Zero()};
        log_T.template block<2,2>(0,0) = SE2::skew(w(0));
        log_T.template block<2,1>(0,2) = w.template segment<2>(1);
        return log_T;
    }

    static Vec3F vee(const Mat3F &log_T)
    {
        Vec3F w;
        w << log_T(1,0), log_T(0,2), log_T(1,2);
        return w;
    }

    Mat3F log() const 
    {
        return SE2::log(this->T());
    }

    static Mat3F log(const SE2 &T)
    {
        return T.log();
    }

    static Mat3F log(const Mat3F &T)
    {
        F theta{atan2(T(1,0), T(0,0))};
        transF t{T.template block<2,1>(0,2)};

        F A, B;
        if(abs(theta) > 1e-6)
        {
            A = sin(theta) / theta;
            B = (1 - cos(theta)) / theta;
        }
        else 
        {
            A = 1.0 - pow(theta,2) / 6.0 + pow(theta, 4) / 120.0;
            B = theta / 2.0 - pow(theta, 3) / 24.0 + pow(theta,5) / 720.0;
        }

        F normalizer{1.0 / (pow(A,2) + pow(B,2))};
        RotF temp;
        temp << A, B, -B, A;
        RotF V_inv{normalizer * temp};

        Mat3F log_T{Mat3F::Zero()};
        log_T.template block<2,1>(0,2) = V_inv * t;
        log_T(0,1) = -theta;
        log_T(1,0) = theta;

        return log_T;
    }

private:
    Mat3F _arr;
};

#endif