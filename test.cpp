#include "factorization_machine/FactorizationMachine.hpp"
#include <Eigen/Eigen>
#include <Eigen/Sparse>
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using Trip = Eigen::Triplet<int>;
using TripVec = std::vector<Trip, Eigen::aligned_allocator<Trip>>;
using SparseMat = Eigen::SparseMatrix<int8_t>;
using YVec = Eigen::VectorXf;
using stdYVec = std::vector<float>;
const std::string fname = "../full_set.txt_shuffle";

int reader(TripVec &vec, stdYVec &y, const int start,  const int maxR = 1e+6) {
  std::ifstream file(fname);
  if (!file) {
    std::cout << "invalid file" << std::endl;
    return 0;
  }
  int u, f, r;
  int i = 0;
  int counter = 0;
  std::cout << start * maxR << std::endl;
  int maxCount = start * maxR;
  for (;;) {
    
    file >> u >> f >> r;
    if(counter < maxCount) {
      counter ++;
      continue;
    }
    vec.push_back(Trip(i, u, 1));
    vec.push_back(Trip(i, f, 1));
    y.push_back(r);
    i++;
    if (maxR != -1)
      if (i >= maxR)
        break;

    if (file.eof())
      break;
  }
  std::cout << counter << std::endl;
  return i;
}

int main() {
  std::ofstream outfile("out.txt");

  const int c = 17770 + 480189;
  FactorizationMachine fm(c);

  for(int k = 0; k < 10; ++k) {
    std::cout << "ITERTION NUM " << k <<std::endl;
  for (int j = 0; j < 90; ++j) {
    std::cout << "Batch num " << j << std::endl;
    TripVec vec;
    stdYVec std_y;

    const int r = reader(vec, std_y, j);
    
    
    SparseMat a(r, c);
    a.setFromTriplets(vec.begin(), vec.end());

    YVec b(std_y.size());
    for (size_t i = 0; i < std_y.size(); ++i)
      b(i) = std_y[i];

    auto start = std::chrono::high_resolution_clock::now();

    float rm = fm.train(a, b, 1);
    outfile << k << " " << j << " " << rm << "\n";
    auto stop = std::chrono::high_resolution_clock::now();
    float time =
        std::chrono::duration_cast<std::chrono::milliseconds>(stop - start)
            .count();
    std::cout << " Time = " << time << std::endl;
      
  }
  }
    TripVec vec;
    stdYVec std_y;

    const int r = reader(vec, std_y, 95);
    SparseMat a(r, c);
    a.setFromTriplets(vec.begin(), vec.end());

    YVec b(std_y.size());
    for (size_t i = 0; i < std_y.size(); ++i)
      b(i) = std_y[i];
 
    auto y_pred = fm.predict(a);
    float rmse = fm.RMSE(y_pred, b);
    std::cout << "FINAL RMSE = " << rmse << std::endl;
    outfile << "final : " << rmse << std::endl; 
  return 0;
}
