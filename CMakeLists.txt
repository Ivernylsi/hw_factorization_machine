project(fm)

cmake_minimum_required(VERSION 3.7)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -std=c++17")

find_package(Eigen3 REQUIRED)

add_subdirectory(factorization_machine)

add_executable(fm_test
               main.cpp
               )

target_link_libraries(fm_test f_machine)
