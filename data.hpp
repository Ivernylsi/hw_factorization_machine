#ifndef DATA_HPP
#define DATA_HPP

#include <Eigen/Eigen>
#include <Eigen/Sparse>
#include <algorithm>
#include <vector>

using Trip = Eigen::Triplet<int>;
using TripVec = std::vector<Trip, Eigen::aligned_allocator<Trip>>;
using SparseMat = Eigen::SparseMatrix<int8_t>;
using YVec = Eigen::VectorXf;
using stdYVec = std::vector<float>;

struct triple {
  int u, f, r;
  triple(int a, int b, int c) : u(a), f(b), r(c) {}
};

struct data {
  std::vector<triple> train;
  std::vector<triple> test;
  data() {}

  void init() {
    curr = 0;
    curr_test = 0;
  }
  void shuffle() { std::random_shuffle(train.begin(), train.end()); }

  bool getNext_train(TripVec &vec, stdYVec &y) {
    vec.clear();
    y.clear();
    if (curr >= static_cast<int>(train.size()) - 1)
      return false;
    int i = 0;
    for (; i < maxIt; ++i) {
      if (i + curr >= (int)train.size() - 1)
        break;
      vec.push_back(Trip(i, train[curr + i].u, 1));
      vec.push_back(Trip(i, train[curr + i].f, 1));
      y.push_back(train[curr + i].r);
    }
    curr += i;
    return true;
  }

  bool getNext_test(TripVec &vec, stdYVec &y) {
    vec.clear();
    y.clear();

    if (curr_test >= static_cast<int>(test.size()) - 1)
      return false;
    int i = 0;
    for (; i < maxIt; ++i) {
      if (i + curr_test >= (int)test.size() - 1)
        break;

      vec.push_back(Trip(i, test[curr_test + i].u, 1));
      vec.push_back(Trip(i, test[curr_test + i].f, 1));
      y.push_back(train[curr_test + i].r);
    }
    curr_test += i;
    return true;
  }

  const int maxIt = 1e+6;
  int curr = 0, curr_test = 0;
};

#endif // DATA_HPP
