#ifndef SE2_H
#define SE2_H

#include <Eigen/Dense>
#include <random>
#include <iostream>
#include <cmath>

#include "utils.h"

namespace rigidTransform
{

template<typename F>
class SE2
{
    using Mat3F = Eigen::Matrix<F,3,3>;
    using Mat2F = Eigen::Matrix<F,2,2>;
    using Vec2F = Eigen::Matrix<F,2,1>;
    using Vec3F = Eigen::Matrix<F,3,1>;
    using Map3F = Eigen::Map<Mat3F>;
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    SE2() : arr_(data_) { arr_.setIdentity(); }

    SE2(const F *data) : arr_(const_cast<F*>(data)) {}

    SE2(const Eigen::Ref<const Mat3F> &T) : arr_(data_) { arr_ = T; }

    SE2(const Eigen::Ref<const Vec2F> &t, const F& theta) : arr_(data_)
    {
        F ct{cos(theta)}, st{sin(theta)};
        arr_ << ct, -st, t(0), st, ct, t(1), 0, 0, 1;
    }

    SE2(const F& x, const F& y, const F& theta) : arr_(data_)
    {
        F ct{cos(theta)}, st{sin(theta)};
        arr_ << ct, -st, x, st, ct, y, 0, 0, 1;
    }

    SE2(const Eigen::Ref<const Mat2F> &R, const Eigen::Ref<const Vec2F> &t) : arr_(data_)
    {
        arr_.setIdentity();
        arr_.template block<2,2>(0,0) = R;
        arr_.template block<2,1>(0,2) = t;
    }

    SE2(const Eigen::Ref<const Mat2F> &R, const F& x, const F& y) : arr_(data_)
    {
        arr_.setIdentity();
        arr_.template block<2,2>(0,0) = R;
        arr_(0,2) = x;
        arr_(1,2) = y;
    }

    SE2(const SE2& T) : arr_(data_) { arr_ = T.T(); }

    SE2& operator=(const SE2& rhs)
    {
        arr_ = rhs.T();
        return (*this);
    }

    SE2 operator*(const SE2& rhs)
    {
        return SE2(T() * rhs.T());
    }

    Mat3F T() const { return arr_; }

    Mat2F R() const { return arr_.template block<2,2>(0,0); }

    Vec2F t() const { return arr_.template block<2,1>(0,2); }

    Mat3F Adj() const
    {
        Mat2F J;
        J << 0, -1, 1, 0;
        Mat3F adj(Mat3F::Identity());
        adj.template block<2,2>(0,0) = R();
        adj.template block<2,1>(0,2) = -J * t();

        return adj;
    }

    SE2 inverse() const
    {
        Mat3F T(Mat3F::Identity());
        T.template block<2,2>(0,0) = R().transpose();
        T.template block<2,1>(0,2) = -R().transpose() * t();
        return SE2(T);
    }

    void inverse_()
    {
        arr_.template block<2,2>(0,0) = R().transpose();
        arr_.template block<2,1>(0,2) = -R() * t();
    }

    Vec2F transa(const Eigen::Ref<const Vec2F> &pt)
    {
        return R() * pt + t();
    }

    Vec2F transp(const Eigen::Ref<const Vec2F> &pt)
    {
        return inverse().transa(pt);
    }

    SE2 boxplusr(const Eigen::Ref<const Vec3F> &tau)
    {
        return (*this) * SE2::Exp(tau);
    }

    Vec3F boxminusr(const SE2 &T)
    {
        return SE2::Log(T.inverse() * (*this));
    }

    SE2 boxplusl(const Eigen::Ref<const Vec3F> &tau)
    {
        return SE2::Exp(tau) * (*this);
    }

    Vec3F boxminusl(const SE2 &T)
    {
        return SE2::Log((*this) * T.inverse());
    }

    F* data()
    {
        return arr_.data();
    }

    static SE2 random()
    {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_real_distribution<F> r_dist(-PI, PI);
        static std::uniform_real_distribution<F> t_dist(-10.0, 10.0);

        F theta{r_dist(gen)};
        F x{t_dist(gen)}, y{t_dist(gen)};
        return SE2(x, y, theta);
    }

    static SE2 Identity()
    {
        return SE2();
    }

    Vec3F Log() const
    {
        return SE2::Log((*this));
    }

    static Vec3F Log(const SE2& T)
    {
        F theta{atan2(T.T()(1,0), T.T()(0,0))}; // should I add () to index??
        Vec2F t(T.t());

        F A,B;
        if(abs(theta) > 1e-8)
        {
            A = sin(theta)/theta;
            B = (1 - cos(theta))/theta;
        }
        else
        {
            A = F(1.0);
            B = F(theta/2.0);
        }
        F normalizer{1.0/(A*A + B*B)};
        Mat2F temp_arr;
        temp_arr << A, B, -B, A;
        Mat2F V_inv = temp_arr * normalizer;

        Vec3F logT;
        logT << V_inv * t, theta;
        return logT;
    }

    static SE2 Exp(const Eigen::Ref<const Vec3F> &logT)
    {
        F theta(logT(2));
        Vec2F rho(logT.template head<2>());

        F A,B;
        if(abs(theta) > 1e-8)
        {
            A = sin(theta)/theta;
            B = (1 - cos(theta))/theta;
        }
        else
        {
            A = F(1.0);
            B = F(theta/2.0);
        }
        Mat2F V;
        V << A, -B, B, A;
        Vec2F t(V * rho);
        return SE2(t, theta);
    }

private:
    F data_[9];
public:
    Eigen::Map<Mat3F> arr_;
};

template<typename F>
std::ostream& operator <<(std::ostream &os, const SE2<F> &T)
{
    os << T.T();
    return os;
}

} // namespace rigidTransform

#endif
