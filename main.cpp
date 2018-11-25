#include "factorization_machine/FactorizationMachine.hpp"
#include <Eigen/Eigen>
#include <Eigen/Sparse>
#include <iostream>

int main() {

  Eigen::MatrixXd a(5, 4);

  a << 5, 4, 1, 1, -1, 3, -1, 1, -1, 1, -1, -1, -1, -1, 5, 1, 1, 5, 4, 4;
  a  =  a * 10;
  Eigen::SparseMatrix<double> mt(13, 5 + 4);
  Eigen::VectorXd y(13);
  mt.setZero();
  int counter = 0;
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 5; ++j) {
      if (a(j,i) != -10 ) {
        mt.coeffRef(counter, j) = 1;
        mt.coeffRef(counter, 5 + i) = 1;
        y(counter++) = a(j, i);
      }
    }

  FactorizationMachine f(mt.cols(), 4);
  f.train(mt, y);

  return 0;
}
