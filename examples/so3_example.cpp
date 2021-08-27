/*
This is an example using the SO2 class. This class will find the mean of a
bunch of rotation matrices in the plane
*/
#include "so3.h"
#include "utils.h"
#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/MatrixFunctions>

#include <ceres/ceres.h>
#include <random>
#include <vector>
#include <iostream>

namespace rt = rigidTransform;
// Issue with jacobians??

class RotationResidual {

 public:
    RotationResidual(const rt::SO3<double> &R, const Eigen::Matrix3d &cov) :
                                                            R_{R} {
        info_ = cov.inverse().llt().matrixU();
    }

    template <typename T>
    bool operator() (const T* const r, T* residuals) const {
        rt::SO3<T> R(r);
        Eigen::Map<Eigen::Matrix<T, 3, 1>> res(residuals);
        res = info_ * rt::SO3<T>::Log(R.inverse() * R_);
        // std::cout << res << std::endl;
        return true;
    }

 private:
    rt::SO3<double> R_;
    Eigen::Matrix3d info_;
};
using RotationCost = ceres::AutoDiffCostFunction<RotationResidual, 3, 9>;

class SO3_Parameterization {
 public:
    template <typename T>
    bool operator() (const T* rot, const T* delta, T* R_plus_delta) const {
        rt::SO3<T> R(rot), Rpd(R_plus_delta);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> t(delta);
        Eigen::Matrix<T, 3, 3> tx = rt::skew3<T>(t);
        Eigen::Matrix<T, 3, 3> dR = Eigen::Matrix<T, 3, 3>::Identity() + tx;
        rt::SO3<T> Expt = rt::SO3<T>(dR);
        Rpd = R * Expt;
        // Rpd = R.boxplusr(t);
        // std::cout << "----t---\n" << t << std::endl;
        // std::cout << "----tx---\n" << dR << std::endl;
        // std::cout << "----Expt---\n" << Expt << std::endl;
        // std::cout << "----Rpd---\n" << Rpd << std::endl;

        // T theta = t.norm();
        // if (theta != T(0)) {
        //     Eigen::Matrix<T, 3, 3> tx;
        //     tx << T(0), -t(2), t(1), t(2), T(0), -t(0), -t(1), t(0), T(0);
        //     Eigen::Matrix<T, 3, 3> exp =
        //         Eigen::Matrix<T, 3, 3>::Identity() + sin(theta)/theta * tx +
        //         (T(1) - cos(theta))/pow(theta, 2) * tx * tx;
        //     rt::SO3<T> Expt(exp);
        //     Rpd = R * Expt;
        // } else {
        //     Rpd = R;
        // }

        return true;
    }
};

int main(int argc, char *argv[]) {
    // Establish the mean rotation
    rt::SO3<double> R = rt::SO3<double>::random();

    // Setup noise distribution
    double theta_std{3 * rt::PI/180};
    Eigen::Matrix3d cov =
        Eigen::Vector3d{theta_std, theta_std, theta_std}.asDiagonal();
    std::random_device rd;
    std::mt19937 engine(rd());
    std::normal_distribution<> generator(0, theta_std);

    // Generate Noisy Measurements
    Eigen::Vector3d theta;
    size_t num_rotations{100};
    std::vector<rt::SO3<double>> measurements{};
    for (size_t i{0}; i < num_rotations; ++i) {
        theta << generator(engine), generator(engine), generator(engine);
        measurements.emplace_back(R.boxplusr(theta));
    }
    rt::SO3<double> R_hat = measurements[0];
    std::cout << "Initial Guess:\n" << R_hat << std::endl;


    // Setup ceres problem
    ceres::Problem problem;

    // Add the reisduals
    for (rt::SO3<double> m : measurements) {
        ceres::CostFunction *cost = new RotationCost(new
                RotationResidual(m, cov));
        problem.AddResidualBlock(cost, nullptr, R_hat.data());
    }

    // Set the local parameterization
    ceres::LocalParameterization *rot_param = new
            ceres::AutoDiffLocalParameterization<SO3_Parameterization, 9, 3>();
    problem.SetParameterization(R_hat.data(), rot_param);

    // Set solver options
    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = false;

    // Solve problem
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << "True Mean:\n" << R << std::endl;
    std::cout << "Est Mean:\n" << R_hat << std::endl;
    // std::cout << summary.FullReport() << std::endl;

    return 0;
}
