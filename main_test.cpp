#include "factorization_machine/FactorizationMachine.hpp"
#include <Eigen/Eigen>
#include <Eigen/Sparse>
#include <chrono>
#include <iostream>
#include <vector>

using triples = Eigen::Triplet<int8_t>;

int main() {
  int r = 1000;
  int c = 100;
  std::vector<triples> vec;
  
  srand(time(0));  
  std::vector<int8_t> y_vec;

  int counter = 0;
  for (int i = 0; i < r; ++i) {
    for (int j = 0; j < c; ++j) {
      int ran = rand() % 5;
      if(ran == 0) continue;
      vec.push_back(triples(counter, i, 1));
      vec.push_back(triples(counter++, r+j, 1));
      y_vec.push_back(ran);
    }
  }

  Eigen::VectorXf y(y_vec.size(), 1);
  for(size_t i = 0; i< y_vec.size();++i) y(i) = y_vec[i];

  std::cout << vec.size() << std::endl;
  Eigen::SparseMatrix<int8_t> mt(counter, r + c);
  mt.setFromTriplets(vec.begin(), vec.end());

  printf("cols : %ld, rows : %ld\n", mt.cols(), mt.rows());
  FactorizationMachine f(mt.cols());

  auto start = std::chrono::high_resolution_clock::now();
  std::cout << "training started" << std::endl;
  f.train(mt, y, 10000);
  auto stop = std::chrono::high_resolution_clock::now();

  float time =
      std::chrono::duration_cast<std::chrono::milliseconds>(stop - start)
          .count();
  std::cout << " Time = " << time << std::endl;
  return 0;
}
