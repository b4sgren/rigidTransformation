/*
This is an example using the SE2 class. It will run a pose graph optimization
routine using Ceres
*/
#include <ceres/ceres.h>

#include <eigen3/Eigen/Dense>
#include <fstream>
#include <string>
#include <vector>

#include "rigidTransformation/se2.h"
#include "rigidTransformation/utils.h"

namespace rt = rigidTransform;

struct EdgeData {
    EdgeData(const rt::SE2<double> &T, const Eigen::Matrix3d &I,
             size_t from_id, size_t to_id) : T_(T), info_(I), from_id_{from_id}, to_id_{to_id} {}

    rt::SE2<double> T_;
    Eigen::Matrix3d info_;
    size_t from_id_;
    size_t to_id_;
};

class EdgeResidual {
   public:
    EdgeResidual(const rt::SE2<double> &T, const Eigen::Matrix3d &info) : dT_(T) {
        Xi_ = info.llt().matrixU();
    }

    template <typename T>
    bool operator()(const T *const T1, const T *const T2, T *residuals) const {
        rt::SE2<T> T1_(T1), T2_(T2);
        Eigen::Map<Eigen::Matrix<T, 3, 1>> res(residuals);

        rt::SE2<T> dT = T1_.inverse() * T2_;
        res = Xi_ * dT_.template boxminusr<T, T>(dT);

        return true;
    }

    static ceres::CostFunction *Create(rt::SE2<double> &T, Eigen::Matrix3d &info) {
        return new ceres::AutoDiffCostFunction<EdgeResidual, 3, 9, 9>(new EdgeResidual(T, info));
    }

   private:
    rt::SE2<double> dT_;
    Eigen::Matrix3d Xi_;
};

class SE2Manifold {
   public:
    template <typename F>
    bool Plus(const F *transform, const F *delta, F *T_plus_delta) const {
        rt::SE2<F> T(transform), res(T_plus_delta);
        Eigen::Map<const Eigen::Matrix<F, 3, 1>> nu(delta);
        res = T.template boxplusr<F, F>(nu);

        return true;
    }

    template <typename F>
    bool Minus(const F *transform1, const F *transform2, F *diff) const {
        rt::SE2<F> T1(transform1), T2(transform2);
        Eigen::Map<Eigen::Matrix<F, 3, 1>> nu(diff);
        nu = T1.template boxminusr<F, F>(T2);
        // rt::SE2<F> dT = T1.inverse() * T2;
        // nu = rt::SE2<F>::Log(dT);

        return true;
    }

    static ceres::Manifold *Create() {
        return new ceres::AutoDiffManifold<SE2Manifold, 9, 3>;
    }
};

void readData(const std::string &filename, std::vector<rt::SE2<double>> &poses,
              std::vector<EdgeData> &edges) {
    std::ifstream fin;
    fin.open(filename);
    assert(fin.is_open());

    std::string line_id;
    size_t from_id, to_id;
    double dx, dy, dtheta;
    double cxx, cxy, cxt, cyy, cyt, ctt;

    while (fin >> line_id) {
        if (line_id[0] == 'V') {
            fin >> from_id >> dx >> dy >> dtheta;
            poses.emplace_back(dx, dy, dtheta);
        } else {
            fin >> from_id >> to_id >> dx >> dy >> dtheta >> cxx >> cxy >> cxt >> cyy >> cyt >> ctt;
            rt::SE2<double> edge{dx, dy, dtheta};
            Eigen::Matrix3d cov;
            cov << cxx, cxy, cxt, cxy, cyy, cyt, cxt, cyt, ctt;
            edges.emplace_back(edge, cov, from_id, to_id);
        }
    }
    fin.close();
}

int main(int argc, char *argv[]) {
    // Read in the data
    std::string filename{argv[1]};
    std::vector<rt::SE2<double>> poses{};
    std::vector<EdgeData> edges{};
    readData(filename, poses, edges);

    // Setup the Problem
    ceres::Problem problem;
    ceres::Manifold *se2_manifold = SE2Manifold::Create();

    for (EdgeData &e : edges) {
        ceres::CostFunction *cost = EdgeResidual::Create(e.T_, e.info_);
        problem.AddResidualBlock(cost, nullptr, poses[e.from_id_].data(),
                                 poses[e.to_id_].data());
        problem.SetManifold(poses[e.from_id_].data(), se2_manifold);
        problem.SetManifold(poses[e.to_id_].data(), se2_manifold);
    }

    // Set first element constant
    problem.SetParameterBlockConstant(poses[0].data());

    // Solve the problem
    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << std::endl;

    std::ofstream fout{"data.txt"};
    for (rt::SE2<double> p : poses)
        fout << p.t().transpose() << std::endl;
    fout.close();

    return 0;
}
