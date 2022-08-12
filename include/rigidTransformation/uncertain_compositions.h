#ifndef RIGIDTRANSFORMATION_UNCERTAIN_COMPOSITIONS_H_
#define RIGIDTRANSFORMATION_UNCERTAIN_COMPOSITIONS_H_

#include <eigen3/Eigen/Dense>

namespace rigidTransform {

template <typename T>
class RightInvariant {
   public:
    RightInvariant() = default;

    static void inv(const T& Tij, const Eigen::MatrixXd& cov_ij, T& Tji,
                    Eigen::MatrixXd& cov_ji) {
        Tji = Tij.inverse();
        cov_ji = Tij.Adj() * cov_ij * Tij.Adj().transpose();
    }

    // TODO: How to best deal with uncertainty
    static void compose(const T& Tij, const Eigen::MatrixXd& cov_ij,
                        const T& Tjk, const Eigen::MatrixXd& cov_jk, T& Tik,
                        Eigen::MatrixXd& cov_ik) {
        Tik = Tij * Tjk;
        Eigen::MatrixXd Adj_kj = Tjk.inverse().Adj();
        cov_ik = Adj_kj * cov_ij * Adj_kj.transpose() + cov_jk;
    }

    // TODO: How to best deal with uncertainty
    static void between(const T& Tij, const Eigen::MatrixXd& cov_ij,
                        const T& Tik, const Eigen::MatrixXd& cov_ik, T& Tjk,
                        Eigen::MatrixXd& cov_jk) {
        Tjk = Tij.inverse() * Tik;
        Eigen::MatrixXd Adj = Tik.inverse().Adj() * Tij.Adj();
        cov_jk = Adj * cov_ij * Adj.transpose() + cov_ik;
    }

   private:
};

}  // namespace rigidTransform

#endif  // RIGIDTRANSFORMATION_UNCERTAIN_COMPOSITIONS_H_
