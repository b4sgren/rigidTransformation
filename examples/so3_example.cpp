/*
This is an example using the SO2 class. This class will find the mean of a
bunch of rotation matrices in the plane
*/
#include <ceres/autodiff_manifold.h>
#include <ceres/ceres.h>

#include <eigen3/Eigen/Dense>
#include <iostream>
#include <random>
#include <vector>

#include "rigidTransformation/so3.h"
#include "rigidTransformation/utils.h"

namespace rt = rigidTransform;

class RotationResidual {
   public:
    RotationResidual(const rt::SO3<double>& R, const Eigen::Matrix3d& cov) : R_{R} {
        info_ = cov.inverse().llt().matrixU();
    }

    template <typename T>
    bool operator()(const T* const r, T* residuals) const {
        rt::SO3<T> R(r);
        Eigen::Map<Eigen::Matrix<T, 3, 1>> res(residuals);
        res = info_ * rt::SO3<T>::Log(R.inverse() * R_);
        return true;
    }

    static ceres::CostFunction* Create(rt::SO3<double>& R, const Eigen::Matrix3d& cov) {
        return new ceres::AutoDiffCostFunction<RotationResidual, 3, 9>(new RotationResidual(R, cov));
    }

   private:
    rt::SO3<double> R_;
    Eigen::Matrix3d info_;
};

class SO3Manifold {
   public:
    template <typename T>
    bool Plus(const T* rot, const T* delta, T* R_plus_delta) const {
        rt::SO3<T> R(rot), res(R_plus_delta);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> theta(delta);
        res = R.template boxplusr<T>(theta);

        return true;
    }

    template <typename T>
    bool Minus(const T* rot1, const T* rot2, T* diff) const {
        rt::SO3<T> R1(rot1), R2(rot2);
        Eigen::Map<Eigen::Matrix<T, 3, 1>> theta(diff);
        theta = R1.boxminusr(R2);

        return true;
    }

    static ceres::Manifold* Create() {
        return new ceres::AutoDiffManifold<SO3Manifold, 9, 3>;
    }
};

int main(int argc, char* argv[]) {
    // Establish the mean rotation
    rt::SO3<double> R = rt::SO3<double>::random();

    // Setup noise distribution
    double theta_std{3 * rt::PI / 180};
    Eigen::Matrix3d cov =
        Eigen::Vector3d{theta_std, theta_std, theta_std}.asDiagonal();
    std::random_device rd;
    std::mt19937 engine(rd());
    std::normal_distribution<> generator(0, theta_std);

    ceres::Manifold* so3_manifold = SO3Manifold::Create();

    // Generate Noisy Measurements
    Eigen::Vector3d theta;
    size_t num_rotations{100};
    std::vector<rt::SO3<double>> measurements{};
    for (size_t i{0}; i < num_rotations; ++i) {
        theta << generator(engine), generator(engine), generator(engine);
        measurements.emplace_back(R.boxplusr<double>(theta));
    }
    rt::SO3<double> R_hat = measurements[0];
    std::cout << "Initial Guess:\n"
              << R_hat << std::endl;

    // Setup ceres problem
    ceres::Problem problem;

    // Add the reisduals
    for (rt::SO3<double> m : measurements) {
        ceres::CostFunction* cost = RotationResidual::Create(m, cov);
        problem.AddResidualBlock(cost, nullptr, R_hat.data());
    }
    // Set the manifold
    problem.SetManifold(R_hat.data(), so3_manifold);

    // Set solver options
    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;

    // Solve problem
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << "True Mean:\n"
              << R << std::endl;
    std::cout << "Est Mean:\n"
              << R_hat << std::endl;
    std::cout << summary.FullReport() << std::endl;

    return 0;
}
