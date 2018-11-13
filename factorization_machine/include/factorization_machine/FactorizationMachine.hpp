#ifndef FACTORIZATIONMACHINE_HPP
#define FACTORIZATIONMACHINE_HPP
#include <Eigen/Eigen>
#include <Eigen/Sparse>

class FactorizationMachine {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  FactorizationMachine(int n, int k = 2);

  void train(const Eigen::SparseMatrix<double> data, const Eigen::VectorXd &y);

private:
  auto commonCompute(const Eigen::SparseMatrix<double> data, int Idx) -> auto {
    return data * v.col(Idx);
  }
  auto commonSquared(const Eigen::SparseMatrix<double> data, int Idx) -> auto {
    return commonCompute(data, Idx).array().abs2();
  }
  auto second_part(const Eigen::SparseMatrix<double> data) -> auto {
    Eigen::VectorXd ans =
        commonCompute(data, 0).array().abs2() - commonSquared(data, 0);
    for (int i = 1; i < k; ++i)
      ans = ans.array() + commonCompute(data, 0).array().abs2() -
            commonSquared(data, 0);
    return ans;
  }

  double RMSE(const Eigen::VectorXd &y_pred, const Eigen::VectorXd &y);

  double R2(const Eigen::VectorXd &y_pred, const Eigen::VectorXd &y);

  // Gradient block
  double gradientW0();
  Eigen::VectorXd gradientW(const Eigen::VectorXd &x);
  Eigen::MatrixXd gradientV(const Eigen::VectorXd &x,
                            const Eigen::VectorXd &comSq);

  Eigen::MatrixXd v;
  Eigen::VectorXd w;
  double w0;
  int n, k;

public:
  auto predict(const Eigen::SparseMatrix<double> data) -> auto {
    assert(data.cols() == n);
    return w0 + (data * w).array() + second_part(data).sum();
  }
};

#endif // FACTORIZATIONMACHINE_HPP
