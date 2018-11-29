#include "factorization_machine/FactorizationMachine.hpp"
#include <chrono>
#include <iostream>
#include <vector>

FactorizationMachine::FactorizationMachine(int n, int k) : n(n), k(k) {
  v = Eigen::MatrixXf::Random(n, k) / 100;
  w = Eigen::VectorXf::Random(n);
  w0 = 0;
}

void FactorizationMachine::train(const Eigen::SparseMatrix<int8_t> data,
                                 const Eigen::VectorXf &y,
                                 const int max_iteration) {

  common.resize(data.rows(), k);

  float train_rate = 0.1;
  float v_train = 0.1;
  Eigen::VectorXf y_pred = predict(data);

  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < max_iteration; ++i) {
    Eigen::VectorXf diff_y = (y_pred - y);
    //    printf("diff is %f\n", diff_y.array().abs().sum());

    w0 -= diff_y.mean() * train_rate;

    auto t = diff_y.transpose() * data.cast<float>() / y.rows();

    w -= train_rate * t.transpose();

    for (int f = 0; f < k; ++f) {
      Eigen::VectorXf a1 =
          common.col(f).transpose() * data.cast<float>() / y.rows();

      auto Vf = (data.cast<float>() * v.col(f).asDiagonal()).transpose();

      Eigen::VectorXf a2 = Vf * Eigen::VectorXf::Ones(1, Vf.cols());
      v.col(f) -= (a1 - a2) * v_train;
    }

    if (i % 1000 == 0)
      v_train /= 10;

    if (i % 1 == 0) {
      printf("iteration N %i \n", i);
      printf("diff is %f\n", diff_y.array().abs().mean());
    }

    y_pred = predict(data);
  }

    auto stop = std::chrono::high_resolution_clock::now();
  
  float time = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
  std::cout << "Elapsed Time = "<< time <<std::endl;

  std::cout << "RMSE : " << RMSE(y_pred, y) << std::endl;
}

float FactorizationMachine::RMSE(const Eigen::VectorXf &y_pred,
                                 const Eigen::VectorXf &y) {

  auto diff = (y - y_pred).array().abs2();
  return sqrt(diff.sum() / y.rows());
}

