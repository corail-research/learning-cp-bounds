cmake_minimum_required(VERSION 3.0)
project(solver)

set(pybind11_DIR /Users/dariusdabert/anaconda3/share/cmake/pybind11)
find_package(pybind11 REQUIRED)  # or `add_subdirectory(pybind11)`

SET(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_MODULE_PATH})
find_package(Gecode 6.2 COMPONENTS Driver Float Int Kernel Minimodel Search Set Support)
find_package(Threads)
find_package(Torch REQUIRED)

add_executable(solver_ssp solver_ssp.cpp)
# Link Gecode to target test
target_link_libraries(solver_ssp PRIVATE 
    ${TORCH_LIBRARIES}
    pybind11::embed
    Gecode::Driver
    Gecode::Gist
    Gecode::Minimodel
    Gecode::Int
    Gecode::Float
    Gecode::Set
    Gecode::Kernel
    Gecode::Search
    Gecode::Support
    Threads::Threads)


set_property(TARGET solver_ssp PROPERTY CXX_STANDARD 17)