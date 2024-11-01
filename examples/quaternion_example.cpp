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

#include "rigidTransformation/quaternion.h"
#include "rigidTransformation/utils.h"

namespace rt = rigidTransform;

class RotationResidual {
   public:
    RotationResidual(const rt::Quaternion<double>& q,
                     const Eigen::Matrix3d& cov) : q_{q} {
        info_ = cov.inverse().llt().matrixU();
    }

    template <typename T>
    bool operator()(const T* const r, T* residuals) const {
        rt::Quaternion<T> q(r);
        Eigen::Map<Eigen::Matrix<T, 3, 1>> res(residuals);
        // res = info_ * rt::Quaternion<T>::Log(q.inverse() * q_);
        res = info_ * q_.template boxminusr<T, T>(q);
        return true;
    }

    static ceres::CostFunction* Create(rt::Quaternion<double>& q, Eigen::Matrix3d& cov) {
        return new ceres::AutoDiffCostFunction<RotationResidual, 3, 4>(new RotationResidual(q, cov));
    }

   private:
    rt::Quaternion<double> q_;
    Eigen::Matrix3d info_;
};

class QuatManifold {
   public:
    template <typename T>
    bool Plus(const T* rot, const T* delta, T* q_plus_delta) const {
        rt::Quaternion<T> q(rot), res(q_plus_delta);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> theta(delta);
        res = q.template boxplusr<T, T>(theta);

        return true;
    }

    template <typename T>
    bool Minus(const T* rot1, const T* rot2, T* diff) const {
        rt::Quaternion<T> q1(rot1), q2(rot2);
        Eigen::Map<Eigen::Matrix<T, 3, 1>> dq(diff);
        dq = q1.template boxminusr<T, T>(q2);

        return true;
    }

    static ceres::Manifold* Create() {
        return new ceres::AutoDiffManifold<QuatManifold, 4, 3>;
    }
};

int main(int argc, char* argv[]) {
    // Establish the mean rotation
    rt::Quaternion<double> q = rt::Quaternion<double>::random();

    // Setup noise distribution
    double theta_std{3 * rt::PI / 180};
    Eigen::Matrix3d cov =
        Eigen::Vector3d{theta_std, theta_std, theta_std}.asDiagonal();
    std::random_device rd;
    std::mt19937 engine(rd());
    std::normal_distribution<> generator(0, theta_std);

    ceres::Manifold* quat_manifold = QuatManifold::Create();

    // Generate Noisy Measurements
    Eigen::Vector3d theta;
    size_t num_rotations{100};
    std::vector<rt::Quaternion<double>> measurements{};
    for (size_t i{0}; i < num_rotations; ++i) {
        theta << generator(engine), generator(engine), generator(engine);
        measurements.emplace_back(q.boxplusr<double, double>(theta));
    }
    rt::Quaternion<double> q_hat = measurements[0];
    std::cout << "Initial Guess:\n"
              << q_hat << std::endl;

    // Setup ceres problem
    ceres::Problem problem;

    // Add the reisduals
    for (rt::Quaternion<double> m : measurements) {
        ceres::CostFunction* cost = RotationResidual::Create(m, cov);
        problem.AddResidualBlock(cost, nullptr, q_hat.data());
    }
    // Set the manifold
    problem.SetManifold(q_hat.data(), quat_manifold);

    // Set solver options
    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;

    // Solve problem
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << "True Mean:\n"
              << q << std::endl;
    std::cout << "Est Mean:\n"
              << q_hat << std::endl;
    std::cout << summary.FullReport() << std::endl;

    return 0;
}
