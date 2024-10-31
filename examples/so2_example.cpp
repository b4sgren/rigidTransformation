/*
This is an example using the SO2 class. This class will find the mean of a
bunch of rotation matrices in the plane
*/
#include <ceres/autodiff_manifold.h>
#include <ceres/ceres.h>

#include <iostream>
#include <random>
#include <vector>

#include "rigidTransformation/so2.h"
#include "rigidTransformation/utils.h"

namespace rt = rigidTransform;

class RotationResidual {
   public:
    RotationResidual(const rt::SO2<double>& R, double cov) : R_{R},
                                                             info_{1.0 / cov} {}

    template <typename T>
    bool operator()(const T* const r, T* residuals) const {
        rt::SO2<T> R(r);
        *residuals = info_ * R_.boxminusr<T, T>(R);
        return true;
    }

    static ceres::CostFunction* Create(rt::SO2<double>& R, double cov) {
        return new ceres::AutoDiffCostFunction<RotationResidual, 1, 4>(new RotationResidual(R, cov));
    }

   private:
    rt::SO2<double> R_;
    double info_;
};

class SO2Manifold {
   public:
    template <typename T>
    bool Plus(const T* rot, const T* delta, T* R_plus_delta) const {
        rt::SO2<T> R(rot), res(R_plus_delta);
        res = R.boxplusr(*delta);

        return true;
    }

    template <typename T>
    bool Minus(const T* rot1, const T* rot2, T* diff) const {
        rt::SO2<T> R1(rot1), R2(rot2);
        (*diff) = R1.template boxminusr<T, T>(R2);

        return true;
    }

    static ceres::Manifold* Create() {
        // <Manifold, GLobal size, Tangent size>
        return new ceres::AutoDiffManifold<SO2Manifold, 4, 1>;
    }
};

int main(int argc, char* argv[]) {
    // Establish the mean rotation
    rt::SO2<double> R = rt::SO2<double>::random();
    // Random Initial Guess
    rt::SO2<double> R_hat = rt::SO2<double>::random();
    std::cout << "Initial Guess:\n"
              << R_hat << std::endl;

    ceres::Manifold* so2_manifold = SO2Manifold::Create();

    // Setup noise distribution
    double theta_std{3 * rt::PI / 180};
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
        ceres::CostFunction* cost = RotationResidual::Create(m, theta_std * theta_std);
        problem.AddResidualBlock(cost, nullptr, R_hat.data());
    }
    // Set the manifold
    problem.SetManifold(R_hat.data(), so2_manifold);

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
