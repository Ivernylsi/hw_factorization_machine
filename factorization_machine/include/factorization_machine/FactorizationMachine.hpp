#ifndef FACTORIZATIONMACHINE_HPP
#define FACTORIZATIONMACHINE_HPP
#include <Eigen/Eigen>
#include <Eigen/Sparse>
#include <iostream>

class FactorizationMachine {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  FactorizationMachine(int n, int k = 2);

  void train_py(Eigen::SparseMatrix<double> data, const Eigen::VectorXd &y,
             const int max_iteration = 1e+4) {
    train(data.cast<int8_t>(), y.cast<float>(), max_iteration); 
  }

  void train(const Eigen::SparseMatrix<int8_t> &data, const Eigen::VectorXf &y,
             const double factor = 1.0);
  float RMSE(const Eigen::VectorXf &y_pred, const Eigen::VectorXf &y);

  int getN() const { return n; }
  int getK() const { return k; }

private:
  auto commonCompute(const Eigen::SparseMatrix<int8_t> &data) -> void {
    common = data.cast<float>() * v;
  }

  auto commonSquared(const Eigen::SparseMatrix<int8_t> &data) -> auto {
    Eigen::MatrixXf v2 = v.array().abs2();
    return data.cast<float>() * v2;
  }

  auto second_part(const Eigen::SparseMatrix<int8_t> &data) -> Eigen::VectorXf {
    commonCompute(data);

    Eigen::MatrixXf comSq = commonSquared(data);
    Eigen::MatrixXf temp = common.array().abs2() - comSq.array();

    Eigen::VectorXf one(k);
    one.setOnes();
    auto ans = temp * one;
    return ans;
  }

  Eigen::MatrixXf v;
  Eigen::VectorXf w;
  float w0;
  int n, k;
  Eigen::MatrixXf common;

public:
  auto predict(const Eigen::SparseMatrix<int8_t> &data) -> Eigen::VectorXf {
    assert(data.cols() == n);
    auto s_part = second_part(data).array() / 2;
    return w0 + (data.cast<float>() * w).array() + s_part;
  }
};

#endif // FACTORIZATIONMACHINE_HPP
