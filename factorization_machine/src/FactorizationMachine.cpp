#include "factorization_machine/FactorizationMachine.hpp"
#include <chrono>
#include <iostream>
#include <random>
#include <vector>

namespace {
float getUniform() {
  std::normal_distribution<float> runif(-1.0, 1.0);
  std::mt19937 rng;
  float ans = runif(rng);
  return ans;
}
} // namespace
FactorizationMachine::FactorizationMachine(int n, int k) : n(n), k(k) {
  auto a = [](const auto &) { return getUniform(); };
  v = Eigen::MatrixXf(n, k);
  v.unaryExpr(a);
  v /= 100;
  w = Eigen::VectorXf(n);
  w.unaryExpr(a);

  w0 = getUniform();
}

float FactorizationMachine::train(const Eigen::SparseMatrix<int8_t> &data,
                                  const Eigen::VectorXf &y,
                                  const int max_iteration) {

  common.resize(data.rows(), k);

  float train_rate = 0.1;
  float v_train = 0.01;
  Eigen::VectorXf y_pred = predict(data);
  float rmse;
  for (int i = 0; i < max_iteration; ++i) {

    Eigen::VectorXf diff_y = (y_pred - y);
    //  printf("diff is %f\n", diff_y.array().abs().sum());

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

    if (i % 1000 == 0)
      v_train /= 2;
  }

  y_pred = predict(data);

  rmse = RMSE(y_pred, y);

  std::cout << "RMSE : " << rmse << std::endl;
  return rmse;
}

float FactorizationMachine::RMSE(const Eigen::VectorXf &y_pred,
                                 const Eigen::VectorXf &y) {

  auto diff = (y - y_pred).array().abs2();
  return sqrt(diff.sum() / y.rows());
}
