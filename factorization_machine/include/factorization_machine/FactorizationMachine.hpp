#ifndef FACTORIZATIONMACHINE_HPP
#define FACTORIZATIONMACHINE_HPP
#include <Eigen/Eigen>
#include <Eigen/Sparse>
#include <iostream>

class FactorizationMachine {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  FactorizationMachine(int n, int k = 2);

  void train(const Eigen::SparseMatrix<double> data, const Eigen::VectorXd &y);

private:
  auto commonCompute(const Eigen::SparseMatrix<double> data) -> void {
    common = data * v;
  }

  auto commonSquared(const Eigen::SparseMatrix<double> data) -> auto {
    Eigen::MatrixXd v2 = v.array().abs2();
    return data * v2;
  }

  auto second_part(const Eigen::SparseMatrix<double> data) -> Eigen::VectorXd {
    commonCompute(data);

    Eigen::MatrixXd comSq = commonSquared(data);

    Eigen::MatrixXd temp = common.array().abs2() - comSq.array();
    
    Eigen::VectorXd one(k); 
    one.setOnes();
    auto ans = temp * one;
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
  Eigen::MatrixXd common;

public:
  auto predict(const Eigen::SparseMatrix<double> data) -> Eigen::VectorXd {
    assert(data.cols() == n);
    auto s_part = second_part(data).array() / 2;
    return w0 + (data * w).array() + s_part;
  }
};

#endif // FACTORIZATIONMACHINE_HPP
