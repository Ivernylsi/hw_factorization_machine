#include "factorization_machine/FactorizationMachine.hpp"
#include <Eigen/Eigen>
#include <Eigen/Sparse>
#include <iostream>

int main() {

  Eigen::MatrixXd a = Eigen::MatrixXd::Random(90, 20);
  
  int r = 90;
  int c = 20;

  Eigen::SparseMatrix<double> mt(r*c, r+c);
  Eigen::VectorXd y(r*c);
  mt.setZero();
  int counter = 0;
  for (int i = 0; i < c; ++i)
    for (int j = 0; j < r; ++j) {
        mt.coeffRef(counter, j) = 1;
        mt.coeffRef(counter, r + i) = 1;
        y(counter++) = a(j, i);
    }

  FactorizationMachine f(mt.cols());
  f.train(mt, y);

  return 0;
}
