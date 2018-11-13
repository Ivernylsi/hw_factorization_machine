project(f_machine)

add_library(f_machine
            src/FactorizationMachine.cpp
            )

target_include_directories(f_machine PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)

target_link_libraries(f_machine Eigen3::Eigen)

