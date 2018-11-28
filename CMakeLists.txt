project(fm)

cmake_minimum_required(VERSION 3.12)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules/")


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -fPIC --pedantic -O3 -Wall -Wextra -std=c++17")


find_package(Python3Interp REQUIRED)
find_package(Eigen3 REQUIRED)

add_subdirectory(factorization_machine)
add_subdirectory(dependencies/pybind11)
add_subdirectory(wrappers)

add_executable(fm_test
               main.cpp
               )

target_link_libraries(fm_test f_machine)
