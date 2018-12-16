#ifndef DATAREADER_HPP
#define DATAREADER_HPP 

#include <Eigen/Eigen>
#include <Eigen/Sparse>
#include <vector>
#include <fstream>
#include <string>
using Trip = Eigen::Triplet<int>;
using TripVec = std::vector<Trip, Eigen::aligned_allocator<Trip>>;
using SparseMat = Eigen::SparseMatrix<int8_t>;
using YVec = Eigen::VectorXf;
using stdYVec = std::vector<float>;

struct data_reader {
  std::ifstream file;

  data_reader(const std::string &s, int sk = 0,  int maxR=1e+6):maxR(maxR) {
    file.open(s);
    curr_line = 0;
    if(sk !=0) skip(sk * maxR);
  }

  void skip(int to_skip) {
  int u, f, r;
  int i;
  for (i = 0; i < to_skip; ++i) 
    file >> u >> f >> r;

  }

  int read(TripVec &vec, stdYVec &y) {
  int u, f, r;
  int i;
  for (i = 0; i < maxR; ++i) {
    file >> u >> f >> r;
    vec.push_back(Trip(i, u, 1));
    vec.push_back(Trip(i, f, 1));
    y.push_back(r);

    if (file.eof()) 
      break;
  }
  return i;
}


  int curr_line;
  int maxR;
};

#endif // DATAREADER_HPP
