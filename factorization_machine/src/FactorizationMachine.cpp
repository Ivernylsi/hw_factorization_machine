#include "factorization_machine/FactorizationMachine.hpp"
#include <vector>
#include <iostream>

FactorizationMachine::FactorizationMachine(int n, int k) : n(n), k(k) {
  v = Eigen::MatrixXd(n, k);
  v.setZero();
  w = Eigen::VectorXd(n);
  w.setZero();
  w0 = 0;
}

void FactorizationMachine::train(const Eigen::SparseMatrix<double> data,
                                 const Eigen::VectorXd &y) {
  int max_iteration = 1e+4;
  double train_rate = 0.001;
  for (int i = 0; i < max_iteration; ++i) {
    Eigen::VectorXd y_pred = predict(data);
    Eigen::VectorXd diff_y = (y_pred - y) / y.rows();
    printf("diff is %f\n", diff_y.sum());
    w0 += diff_y.sum() * train_rate;
    for (int j = 0; j < data.rows(); ++j) {
      Eigen::VectorXd temp = data.row(k);
      w = w.array() - temp.array() * diff_y.array() / y.rows() * train_rate;
    }
      
    for(int f = 0; f < k; ++f) {
      auto a1 = (data * v.col(f)) * data / y.rows();
      auto a2 = data* v.col(f).transpose() / y.rows();
      auto a_d = a1-a2;
      v.col(f) = v.col(f) - a_d * train_rate;
    }

    printf("iteration N %i \n", i);
  }
  std::cout<< "RMSE : " << RMSE(data, y)<<std::endl;;
  std::cout<< "R2 : " << R2(data, y)<<std::endl;;

}

double FactorizationMachine::gradientW0() { return 1; }

Eigen::VectorXd FactorizationMachine::gradientW(const Eigen::VectorXd &x) {
  return x;
}

Eigen::MatrixXd FactorizationMachine::gradientV(const Eigen::VectorXd &x,
                                                const Eigen::VectorXd &com) {
  Eigen::MatrixXd ans(n, k);
  ans.setZero();

  for (int i = 0; i < n; ++i)
    for (int f = 0; f < k; ++f)
      ans(i, f) += x(i) * com(f) - v(i, f) * x(i) * x(i);
  return ans;
}
double FactorizationMachine::RMSE(const Eigen::VectorXd &y_pred,
                                  const Eigen::VectorXd &y) {
  return sqrt((y.array() - y_pred.array()).abs2().sum() / y.rows());
}

double FactorizationMachine::R2(const Eigen::VectorXd &y_pred,
                                const Eigen::VectorXd &y) {

  return 1 - (y.array() - y_pred.array()).abs2().sum() /
                 (y.array() - y.mean()).abs2().sum();
}
