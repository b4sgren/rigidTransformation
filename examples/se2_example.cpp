/*
This is an example using the SE2 class. It will run a pose graph optimization
routine using Ceres
*/
#include "se2.h"
#include "utils.h"

#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>
#include <fstream>
#include <string>
#include <vector>

namespace rt = rigidTransform;

struct EdgeData {
  EdgeData(const rt::SE2<double> &T, const Eigen::Matrix3d &I,
           size_t from_id, size_t to_id) : T_(T), info_(I),
                                           from_id_{from_id}, to_id_{to_id} {}

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
  bool operator() (const T* const T1, const T* const T2, T* residuals) const {
    return true;
  }

 private:
  rt::SE2<double> dT_;
  Eigen::Matrix3d Xi_;
};
using EdgeCost = ceres::AutoDiffCostFunction<EdgeResidual, 3, 9, 9>;

class SE2_Parameterization {
 public:
  template <typename T>
  bool operator() (const T* P, const T* delta, T* T_plus_delta) const {
    return true;
  }
};

void readData(const std::string &filename, std::vector<rt::SE2<double>> &poses,
              std::vector<EdgeData> &edges) {
  std::ifstream fin{filename};
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
      fin >> from_id >> to_id >> dx >> dy >> dtheta >> cxx >> cxy >> cxt >> cyy
          >> cyt >> ctt;
      rt::SE2<double> edge{dx, dy, dtheta};
      Eigen::Matrix3d cov;
      cov << cxx, cxy, cxt, cxy, cyy, cyt, cxt, cyt, ctt;
      edges.emplace_back(edge, cov, from_id, to_id);
    }
  }
}

int main(int argc, char *argv[]) {
  // Read in the data
  std::string filename{argv[0]};
  std::vector<rt::SE2<double>> poses{};
  std::vector<EdgeData> edges{};
  readData(filename, poses, edges);

  // Setup the Problem
  ceres::Problem problem;

  for (EdgeData e : edges) {
    ceres::CostFunction *cost = new EdgeCost(new EdgeResidual(e.T_, e.info_));
    problem.AddResidualBlock(cost, nullptr, poses[e.from_id_].data(),
                             poses[e.to_id_].data());
  }

  ceres::LocalParameterization *param = new
    ceres::AutoDiffLocalParameterization<SE2_Parameterization, 9, 3>();
  for (size_t i{0}; i != poses.size(); ++i) {
    problem.SetParameterization(poses[i].data(), param);
  }

  // Solve the problem
  ceres::Solver::Options options;
  options.minimizer_progress_to_stdout = true;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << std::endl;

  return 0;
}
