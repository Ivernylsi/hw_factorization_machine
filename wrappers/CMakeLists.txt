pybind11_add_module(py_fm
                    PyFactorizationMachine.cpp
                    )

target_link_libraries(py_fm PUBLIC f_machine)
