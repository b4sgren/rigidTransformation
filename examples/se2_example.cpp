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
  EdgeData(const rt::SE2<double> &T, const Eigen::Matrix3d &cov,
           size_t from_id, size_t to_id) : T_(T), cov_(cov),
                                           from_id_{from_id}, to_id_{to_id} {}

  rt::SE2<double> T_;
  Eigen::Matrix3d cov_;
  size_t from_id_;
  size_t to_id_;
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
  // Solve the problem
  return 0;
}
