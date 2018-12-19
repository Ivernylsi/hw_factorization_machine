#ifndef DATAREADER_HPP
#define DATAREADER_HPP

#include <Eigen/Eigen>
#include <Eigen/Sparse>
#include <fstream>
#include <string>
#include <vector>

#include "data.hpp"

struct data_reader {
  std::ifstream file;

  data_reader(const std::string &s) { file.open(s); }

  void read(data &d, const int start) {
    int u, f, r;
    int i = 0;
    const int step = 1e+6 * 10;
    for (;;) {
      file >> u >> f >> r;
      if (i >= (start - 1) * step && i <= start * step)
        d.test.emplace_back(u, f, r);
      else
        d.train.emplace_back(u, f, r);
      if (file.eof())
        break;
      ++i;
    }
    file.close();
  }
};

#endif // DATAREADER_HPP
