#include "factorization_machine/FactorizationMachine.hpp"
#include <iostream>
#include <vector>

FactorizationMachine::FactorizationMachine(int n, int k) : n(n), k(k) {
  v = Eigen::MatrixXd::Random(n, k) / 100;
  w = Eigen::VectorXd::Random(n);
  w0 = 0;
}

void FactorizationMachine::train(const Eigen::SparseMatrix<double> data,
                                 const Eigen::VectorXd &y) {

  common = Eigen::MatrixXd::Zero(data.rows(), k);

  int max_iteration = 1e+4;
  double train_rate = 0.1;
  double v_train = 0.1;
  for (int i = 0; i < max_iteration; ++i) {
    Eigen::VectorXd y_pred = predict(data);
    Eigen::VectorXd diff_y = (y_pred - y);
    //    printf("diff is %f\n", diff_y.array().abs().sum());

    w0 -= diff_y.mean() * train_rate;

    auto t = diff_y.transpose() * data / y.rows();

    w -= train_rate * t.transpose();

    for (int f = 0; f < k; ++f) {
      Eigen::VectorXd a1 = common.col(f).transpose() * data / y.rows();
      Eigen::VectorXd a2 = Eigen::VectorXd::Zero(n);

      // i coulndt avoid this cycle
      for (int jj = 0; jj < data.rows(); ++jj) {
        Eigen::VectorXd temp = data.row(jj).transpose();

        Eigen::VectorXd temp1 = (temp.array() * v.col(f).array());

        a2 += temp1 / y.rows();
      }
      auto a_d = a1 - a2;
      v.col(f) -= a_d * v_train;
    }
    if (i % 3000 == 0)
      v_train /= 10;

    if (i % 10 == 0) {
      printf("iteration N %i \n", i);
      printf("diff is %f\n", diff_y.array().abs().sum());
    }
  }
  Eigen::VectorXd y_pred = predict(data);
  std::cout << "RMSE : " << RMSE(y_pred, y) << std::endl;

  std::cout << "R2 : " << R2(y_pred, y) << std::endl;
}

double FactorizationMachine::RMSE(const Eigen::VectorXd &y_pred,
                                  const Eigen::VectorXd &y) {

  auto diff = (y - y_pred).array().abs2();
  return sqrt(diff.sum() / y.rows());
}

double FactorizationMachine::R2(const Eigen::VectorXd &y_pred,
                                const Eigen::VectorXd &y) {
  double coeff = ((y.array() - y_pred.array()).abs2().sum() /
                  (y.array() - y.mean()).abs2().sum());

  return (1 - coeff);
}
