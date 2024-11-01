/*
This is an example using the SE3 class. It will run a pose graph optimization
routine using Ceres
*/
#include <ceres/ceres.h>

#include <eigen3/Eigen/Dense>
#include <fstream>
#include <string>
#include <vector>

#include "rigidTransformation/se3.h"
#include "rigidTransformation/utils.h"

namespace rt = rigidTransform;
using Matrix6d = Eigen::Matrix<double, 6, 6>;

struct EdgeData {
    EdgeData(const rt::SE3<double> &T, const Matrix6d &I, size_t from_id,
             size_t to_id)
        : T_(T), info_(I), from_id_{from_id}, to_id_{to_id} {}

    rt::SE3<double> T_;
    Matrix6d info_;
    size_t from_id_;
    size_t to_id_;
};

class EdgeResidual {
   public:
    EdgeResidual(const rt::SE3<double> &T, const Matrix6d &info) : dT_(T) {
        Xi_ = info.llt().matrixU();
    }

    template <typename T>
    bool operator()(const T *const T1, const T *const T2, T *residuals) const {
        rt::SE3<T> T1_(T1), T2_(T2);
        Eigen::Map<Eigen::Matrix<T, 6, 1>> res(residuals);

        rt::SE3<T> dT = T1_.inverse() * T2_;
        res = Xi_ * dT_.template boxminusr<T, T>(dT);

        return true;
    }

    static ceres::CostFunction *Create(rt::SE3<double> &T, Matrix6d &info) {
        return new ceres::AutoDiffCostFunction<EdgeResidual, 6, 7, 7>(new EdgeResidual(T, info));
    }

   private:
    rt::SE3<double> dT_;
    Matrix6d Xi_;
};

class SE3Manifold {
   public:
    template <typename T>
    bool Plus(const T *T1, const T *delta, T *T_plus_delta) const {
        rt::SE3<T> T1_(T1), Tpd(T_plus_delta);
        Eigen::Map<const Eigen::Matrix<T, 6, 1>> eta(delta);
        Tpd = T1_.template boxplusr<T, T>(eta);

        return true;
    }

    template <typename T>
    bool Minus(const T *T1, const T *T2, T *diff) const {
        rt::SE3<T> T1_(T1), T2_(T2);
        Eigen::Map<Eigen::Matrix<T, 6, 1>> eta(diff);
        eta = T1_.template boxminusr<T, T>(T2_);

        return true;
    }

    static ceres::Manifold *Create() {
        return new ceres::AutoDiffManifold<SE3Manifold, 7, 6>;
    }
};

void readData(const std::string &filename, std::vector<rt::SE3<double>> &poses,
              std::vector<EdgeData> &edges) {
    std::ifstream fin;
    fin.open(filename);
    assert(fin.is_open());

    std::string line_id;
    size_t from_id, to_id;
    double dx, dy, dz;
    double qx, qy, qz, qw;
    double cxx, cxy, cxz, cxq1, cxq2, cxq3;
    double cyy, cyz, cyq1, cyq2, cyq3;
    double czz, czq1, czq2, czq3;
    double cq1, cq12, cq13, cq2, cq23, cq3;

    while (fin >> line_id) {
        if (line_id[0] == 'V') {
            fin >> from_id >> dx >> dy >> dz >> qx >> qy >> qz >> qw;
            Eigen::Matrix<double, 7, 1> T;
            T << dx, dy, dz, qw, qx, qy, qz;
            poses.emplace_back(T);
            // poses.emplace_back(dx, dy, dz, qw, qx, qy, qz);
        } else {
            fin >> from_id >> to_id >> dx >> dy >> dz >> qx >> qy >> qz >> qw >>
                cxx >> cxy >> cxz >> cxq1 >> cxq2 >> cxq3 >> cyy >> cyz >>
                cyq1 >> cyq2 >> cyq3 >> czz >> czq1 >> czq2 >> czq3 >> cq1 >>
                cq12 >> cq13 >> cq2 >> cq23 >> cq3;
            Eigen::Matrix<double, 7, 1> T;
            T << dx, dy, dz, qw, qx, qy, qz;
            rt::SE3<double> edge(T);  // Doesn't work.
            // rt::SE3<double> edge(dx, dy, dz, qw, qz, qy, qz);
            Matrix6d info;
            info << cxx, cxy, cxz, cxq1, cxq2, cxq3, cxy, cyy, cyz, cyq1, cyq2,
                cyq3, cxz, cyz, czz, czq1, czq2, czq3, cxq1, cyq1, czq1, cq1,
                cq12, cq13, cxq2, cyq2, czq2, cq12, cq2, cq23, cxq3, cyq3, czq3,
                cq13, cq23, cq3;
            edges.emplace_back(edge, info, from_id, to_id);
        }
    }
    fin.close();
}

int main(int argc, char *argv[]) {
    // Read in the data
    std::string filename{argv[1]};
    std::vector<rt::SE3<double>> poses{};
    std::vector<EdgeData> edges{};
    readData(filename, poses, edges);

    // Setup the Problem
    ceres::Problem problem;
    ceres::Manifold *se3_manifold = SE3Manifold::Create();

    for (EdgeData e : edges) {
        ceres::CostFunction *cost = EdgeResidual::Create(e.T_, e.info_);
        problem.AddResidualBlock(cost, nullptr, poses[e.from_id_].data(),
                                 poses[e.to_id_].data());
        problem.SetManifold(poses[e.from_id_].data(), se3_manifold);
        problem.SetManifold(poses[e.to_id_].data(), se3_manifold);
    }

    problem.SetParameterBlockConstant(poses[0].data());

    // Solve the problem
    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 100;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << std::endl;

    std::ofstream fout("data.txt");
    for (rt::SE3<double> p : poses) fout << p.t().transpose() << std::endl;
    fout.close();

    return 0;
}
