#include <fstream>
#include <string>
#include <vector>
#include <iostream>
#include <algorithm>

void reader(std::string fname) {
  std::ifstream file(fname);
  if(!file) { 
    std::cout <<"invalid file" << std::endl;
    return;
  }
  std::string s;
  std::vector<std::string> vec;
  for (;;) {
    getline(file, s);
    vec.push_back(s);
    if (file.eof())
      break;
  }
  srand(time(0));
  std::random_shuffle(vec.begin(), vec.end());
  std::ofstream f(fname + "_shuffle");
  for(auto &i: vec) {
    f << i << std::endl;
  }
}

int main(int argc, char **argv) {
  if(argc!= 2) {
    std::cout << "Usage ./shuffle file_name"<<std::endl;
    return 0;
  }
  reader(std::string(argv[1]));
  return 0;
}
