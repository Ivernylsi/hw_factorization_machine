#include "factorization_machine/FactorizationMachine.hpp"
#include <Eigen/Eigen>
#include <Eigen/Sparse>
#include <iostream>
#include <chrono>

int main() {

  Eigen::MatrixXf a = Eigen::MatrixXf::Random(1000, 100);
  
  int r = (int)a.rows();
  int c = (int)a.cols();;

  Eigen::SparseMatrix<int8_t> mt(r*c, r+c);
  Eigen::VectorXf y(r*c);
  int counter = 0;
  for (int i = 0; i < r; ++i){
    for (int j = 0; j < c; ++j) {
        mt.insert(counter, i) = 1;
        mt.insert(counter, r + j) = 1;
        y(counter++) = a(i, j);
    }
  }
  printf("cols : %ld, rows : %ld\n", mt.cols(), mt.rows());
  FactorizationMachine f(mt.cols());

  auto start = std::chrono::high_resolution_clock::now();
  std::cout << "training started" << std::endl;
  f.train(mt, y, 1);
  auto stop = std::chrono::high_resolution_clock::now();
  
  float time = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
  std::cout << " Time = "<< time <<std::endl;
  return 0;
}
