cmake_minimum_required(VERSION 3.10)
project(BC_Tree)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find pybind11
find_package(pybind11 REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Include directories
include_directories(${pybind11_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/methods)

# Add the executable
add_library(bc_tree MODULE python_wrapper_bh.cpp)

# Link libraries
target_link_libraries(bc_tree PRIVATE pybind11::module OpenMP::OpenMP_CXX)

# Set the output directory for the compiled module
set_target_properties(bc_tree PROPERTIES
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
)