cmake_minimum_required(VERSION 3.10)
project(solver)
option(CXX "enable C++ compilation" ON)

if(CXX)
  enable_language(CXX)
endif()

set(PYBIND11_FINDPYTTHON True)
set(pybind11_DIR /home/bessa75/scratch/scratch2/pybind11/share/cmake/pybind11)
find_package(pybind11 REQUIRED)  # or `add_subdirectory(pybind11)`
find_package(Python3 COMPONENTS Development REQUIRED)

message(STATUS "Python3_INCLUDE_DIRS = ${Python3_INCLUDE_DIRS}")
message(STATUS "Python3_LIBRARIES = ${Python3_LIBRARIES}")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Visual Studio compiler with static runtime libraries
if(MSVC AND MT)
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
  set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /MT")
  set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} /MTd")
endif()
find_package(GUROBI REQUIRED)
include_directories(${GUROBI_INCLUDE_DIRS})

SET(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_MODULE_PATH})
find_package(Gecode 6.2 COMPONENTS Driver Float Int Kernel Minimodel Search Set Support)
find_package(Threads)

add_executable(solver_rcpsp main.cpp)

if(CXX)
  set(CMAKE_CXX_STANDARD 11)
  target_link_libraries(solver_rcpsp PRIVATE optimized ${GUROBI_CXX_LIBRARY}
                        debug ${GUROBI_CXX_DEBUG_LIBRARY})
endif()

# Link Gecode to target test
target_include_directories(solver_rcpsp PRIVATE ${Python3_INCLUDE_DIRS})
target_link_libraries(solver_rcpsp PRIVATE 
    ${GUROBI_LIBRARY}
    ${TORCH_LIBRARIES}
    pybind11::embed
    Python3::Python
    Gecode::Driver
    Gecode::Minimodel
    Gecode::Int
    Gecode::Float
    Gecode::Set
    Gecode::Kernel
    Gecode::Search
    Gecode::Support
    Threads::Threads)


set_property(TARGET solver_rcpsp PROPERTY CXX_STANDARD 17)

if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})
  include(FeatureSummary)
  feature_summary(WHAT ALL)
endif()
