/*
This is an example using the SO2 class. This class will find the mean of a
bunch of rotation matrices in the plane
*/
#include "so2.h"
#include "utils.h"

#include <ceres/ceres.h>
#include <random>
#include <vector>
#include <iostream>

namespace rt = rigidTransform;

class RotationResidual {

 public:
    RotationResidual(const rt::SO2<double> &R, double cov) : R_{R},
                                                             info_{1.0/cov} {}

    template <typename T>
    bool operator() (const T* const r, T* residuals) const {
        rt::SO2<T> R(r);
        *residuals = info_ * rt::SO2<T>::Log(R.inverse() * R_);
        return true;
    }

 private:
    rt::SO2<double> R_;
    double info_;
};
using RotationCost = ceres::AutoDiffCostFunction<RotationResidual, 1, 4>;

class SO2_Parameterization {
 public:
    template <typename T>
    bool operator() (const T* rot, const T* delta, T* R_plus_delta) const {
        rt::SO2<T> R(rot), Rpd(R_plus_delta);
        Rpd = R.boxplusr(*delta);

        return true;
    }
};

int main(int argc, char *argv[]) {
    // Establish the mean rotation
    rt::SO2<double> R = rt::SO2<double>::random();
    // Random Initial Guess
    rt::SO2<double> R_hat = rt::SO2<double>::random();
    std::cout << "Initial Guess:\n" << R_hat << std::endl;

    // Setup noise distribution
    double theta_std{3 * rt::PI/180};
    std::random_device rd;
    std::mt19937 engine(rd());
    std::normal_distribution<> generator(0, theta_std);

    // Generate Noisy Measurements
    size_t num_rotations{100};
    std::vector<rt::SO2<double>> measurements{};
    for (size_t i{0}; i < num_rotations; ++i)
        measurements.emplace_back(R.boxplusr(generator(engine)));

    // Setup ceres problem
    ceres::Problem problem;

    // Add the reisduals
    for (rt::SO2<double> m : measurements) {
        ceres::CostFunction *cost = new RotationCost(new
                RotationResidual(m, theta_std*theta_std));
        problem.AddResidualBlock(cost, nullptr, R_hat.data());
    }

    // Set the local parameterization
    ceres::LocalParameterization *rot_param = new
            ceres::AutoDiffLocalParameterization<SO2_Parameterization, 4, 1>();
    problem.SetParameterization(R_hat.data(), rot_param);

    // Set solver options
    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;

    // Solve problem
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << "True Mean:\n" << R << std::endl;
    std::cout << "Est Mean:\n" << R_hat << std::endl;
    std::cout << summary.FullReport() << std::endl;

    return 0;
}
