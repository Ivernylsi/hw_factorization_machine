#include "factorization_machine/FactorizationMachine.hpp"
#include <chrono>
#include <iostream>
#include <random>
#include <vector>

namespace {
float getUniform() {
  std::normal_distribution<float> runif(0.0, 1.0);
  std::mt19937 rng;
  float ans = runif(rng);
  return ans;
}
} // namespace
FactorizationMachine::FactorizationMachine(int n, int k) : n(n), k(k) {
  auto a = [](const auto &) { return getUniform(); };
  v = Eigen::MatrixXf(n, k);
  v.setZero();
  v = v.unaryExpr(a);
  w = Eigen::VectorXf(n);
  w.setZero();
  w = w.unaryExpr(a);

  w0 = getUniform();
}

void FactorizationMachine::train(const Eigen::SparseMatrix<int8_t> &data,
                                 const Eigen::VectorXf &y,
                                 const double factor) {

  common.resize(data.rows(), k);

  float train_rate = 0.1 * factor;
  float v_train = 0.01  * factor;
  Eigen::VectorXf y_pred = predict(data);

  Eigen::VectorXf diff_y = (y_pred - y);

  w0 -= diff_y.mean() * train_rate;

  auto t = diff_y.transpose() * data.cast<float>() / y.rows();

  w -= train_rate * t.transpose();

  for (int f = 0; f < k; ++f) {
    Eigen::VectorXf a1 =
        common.col(f).transpose() * data.cast<float>() / y.rows();

    Eigen::SparseMatrix<float> Vf =
        (data.cast<float>() * v.col(f).asDiagonal()).transpose();

    Eigen::VectorXf a2 = Vf * Eigen::VectorXf::Ones(Vf.cols(), 1);

    v.col(f) -= (a1 - a2) * v_train / y.rows();
  }

  y_pred = predict(data);
  std::cout << "\t\trmse = " << RMSE(y_pred, y) << std::endl;
}

float FactorizationMachine::RMSE(const Eigen::VectorXf &y_pred,
                                 const Eigen::VectorXf &y) {

  auto diff = (y - y_pred).array().abs2();
  return sqrt(diff.sum() / y.rows());
}
