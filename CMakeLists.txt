#-------------------------------------------------------------------------------
# CMake/C++ configuration
#-------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.13.4)
project(MLIR-PIM)

if(POLICY CMP0075)
  cmake_policy(SET CMP0075 NEW)
endif()

if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

# CMP0116: Ninja generators transform `DEPFILE`s from `add_custom_command()`
# New in CMake 3.20. https://cmake.org/cmake/help/latest/policy/CMP0116.html
if(POLICY CMP0116)
  cmake_policy(SET CMP0116 OLD)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# Use system gcc and g++
# Incase you have conda installed on your system
set(CMAKE_C_COMPILER "/usr/bin/gcc")
set(CMAKE_CXX_COMPILER "/usr/bin/g++")

include(FindPackageHandleStandardArgs)

#-------------------------------------------------------------------------------
# MLIR/LLVM configuration
#-------------------------------------------------------------------------------

set(MLIR_DIR llvm-project/build/lib/cmake/mlir)

find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

option(LLVM_BUILD_TOOLS "Build the LLVM tools. If OFF, just generate build targets." ON)

## Avoid undefined reference to `typeinfo for mlir::Pass'
set(NO_RTTI "-fno-rtti")
add_definitions(${NO_RTTI})

#-------------------------------------------------------------------------------
# PIMOPT Configuration
#-------------------------------------------------------------------------------
set(PIMOPT_SOURCE_DIR ${CMAKE_SOURCE_DIR})
set(PIMOPT_INCLUDE_DIR ${PIMOPT_SOURCE_DIR}/include)
set(PIMOPT_BINARY_DIR ${CMAKE_BINARY_DIR})
set(PIMOPT_TOOLS_DIR ${PIMOPT_BINARY_DIR}/bin)
set(PIMOPT_BIN_INCLUDE_DIR ${PIMOPT_BINARY_DIR}/include)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PIMOPT_BINARY_DIR}/bin)

# Add our own directory to the list of search paths for CMake modules -- Add FindGurobi.cmake file
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

# Pimopt include directories
include_directories(${PIMOPT_INCLUDE_DIR})
include_directories(${PIMOPT_BIN_INCLUDE_DIR})

#-------------------------------------------------------------------------------
# Gurobi setup
#-------------------------------------------------------------------------------

find_package(GUROBI)
if (GUROBI_FOUND)
  message(STATUS "Found Gurobi")
  include_directories(${GUROBI_INCLUDE_DIRS})
else()
  message(WARNING "Gurobi wasn't found, the data layout pass for each layer group \
                   that requires solving an MILP will not be built with the \
                   project.")
  add_compile_definitions(PIMOPT_GUROBI_NOT_INSTALLED)
endif()

#-------------------------------------------------------------------------------
# MLIR-PIM Directory setup
#-------------------------------------------------------------------------------
add_subdirectory(${PIMOPT_SOURCE_DIR}/lib)
add_subdirectory(${PIMOPT_SOURCE_DIR}/tools)
