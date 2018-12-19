#include "DataReader.hpp"
#include "factorization_machine/FactorizationMachine.hpp"
#include <Eigen/Eigen>
#include <Eigen/Sparse>
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

const std::string fname = "../full_set.txt_shuffle";
const int c = 17770 + 480189;

double train_rmse(FactorizationMachine &fm, data &d) {
  srand(time(0));
  double rmse = 0;
  int count = 0;
  TripVec vec;
  stdYVec std_y;
  while (d.getNext_train(vec, std_y)) {
    SparseMat a(std_y.size(), c);
    a.setFromTriplets(vec.begin(), vec.end());

    YVec b(std_y.size());
    for (size_t i = 0; i < std_y.size(); ++i)
      b(i) = std_y[i];
    auto y = fm.predict(a);
    rmse += (y.array() - b.array()).abs2().sum();
    count+=b.size();
  }
  return sqrt(rmse / count);
}

double test_rmse(FactorizationMachine &fm, data &d) {
  double rmse = 0;
  int count = 0;
  TripVec vec;
  stdYVec std_y;
  while (d.getNext_test(vec, std_y)) {
    SparseMat a(std_y.size(), c);
    a.setFromTriplets(vec.begin(), vec.end());

    YVec b(std_y.size());
    for (size_t i = 0; i < std_y.size(); ++i)
      b(i) = std_y[i];
    auto y = fm.predict(a);
    rmse += (y.array() - b.array()).abs2().sum();
    count+=b.size();
  }
  return sqrt(rmse / count);
}

int main(int , char **argv) {
  std::string s(argv[1]);
  int start = std::stoi(s);
  FactorizationMachine fm(c);
  std::cout << "Testing part is starting from " << start << std::endl;
  data d;
  data_reader reader(fname);
  std::cout << "Start reading dataset\n";
  reader.read(d, start);
  std::cout << "Dataset readed\n";
  TripVec vec;
  stdYVec std_y;

  double factor = 1.0;
  for (int k = 0; k < 10; ++k) {
    std::cout << "ITERTION NUM " << k << std::endl;
    std::cout <<"\tShuffling data start" << std::endl;
    d.init(); 
    d.shuffle();
    std::cout <<"\tShuffling data finish" << std::endl;

    while (d.getNext_train(vec, std_y)) {
      SparseMat a(std_y.size(), c);
      a.setFromTriplets(vec.begin(), vec.end());
      YVec b(std_y.size());
      for (size_t i = 0; i < std_y.size(); ++i)
        b(i) = std_y[i];

      auto start = std::chrono::high_resolution_clock::now();

      fm.train(a, b, factor);
      auto stop = std::chrono::high_resolution_clock::now();
      float time =
          std::chrono::duration_cast<std::chrono::milliseconds>(stop - start)
              .count();
      std::cout << "\t\tTime = " << time << std::endl;

    }

    factor *= 0.95;
    d.init();
    std::cout << "\tTrain rmse = " << train_rmse(fm, d) << std::endl;;
    std::cout << "\tTest rmse = " << test_rmse(fm, d) << std::endl;;

  }

  return 0;
}
