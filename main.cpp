#include "factorization_machine/FactorizationMachine.hpp"
#include <Eigen/Eigen>
#include <Eigen/Sparse>
#include <iostream>

int main() {

  Eigen::SparseMatrix<double> mt(100, 100);
  mt.setZero();
  mt.coeffRef(50, 50) = 500;
  FactorizationMachine f(mt.cols());

  std::cout << mt.row(50) << std::endl;
  return 0;
}
