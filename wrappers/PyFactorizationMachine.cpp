#include <Eigen/Eigen>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include <factorization_machine/FactorizationMachine.hpp>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <string>

using namespace pybind11::literals;
namespace py = pybind11;

PYBIND11_MODULE(py_fm, m) {
  py::class_<FactorizationMachine>(m, "Fmachine")
      .def(py::init<int, int>())
      .def("train", &FactorizationMachine::train_py)
      .def("predict", &FactorizationMachine::predict)
      .def("RMSE", &FactorizationMachine::RMSE)
      .def("__repr__", [](const FactorizationMachine &f) {
        std::string repr = "<class FactorizationMachine "
                           "n = " +
                           std::to_string(f.getN());
        repr = repr + ", k = " + std::to_string(f.getK());
        repr = repr + ">";
        return repr;
      });
}
