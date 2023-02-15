##===- CMakeLists.txt - buddy-benchmark cmake root ------------*- cmake -*-===//
##
## Configure DL model benchmark build on both x86 and RISC-V.
##
##===----------------------------------------------------------------------===//

cmake_minimum_required(VERSION 3.10)

#-------------------------------------------------------------------------------
# Project setup and globals.
#-------------------------------------------------------------------------------

project(buddy-benchmark LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_FLAGS "-no-pie")
set(CMAKE_C_FLAGS "-no-pie")

#-------------------------------------------------------------------------------
# BUDDY configuration.
#-------------------------------------------------------------------------------

set(BUDDY_MLIR_BUILD_DIR ../../build)

# MLIR binary directory.
set(LLVM_MLIR_BINARY_DIR ${BUDDY_MLIR_BUILD_DIR}/../llvm/build/bin)

if(CMAKE_TOOLCHAIN_FILE STREQUAL ./riscv-toolchain.cmake)
  # Use cross-compiled RISC-V MLIR library.
  set(LLVM_MLIR_LIBRARY_DIR ${BUDDY_MLIR_BUILD_DIR}/../llvm/build-cross-mlir/lib)
else()
  # Use compiled x86 MLIR library.
  set(LLVM_MLIR_LIBRARY_DIR ${BUDDY_MLIR_BUILD_DIR}/../llvm/build/lib)
endif()

#-------------------------------------------------------------------------------
# Find OpenCV.
#-------------------------------------------------------------------------------

find_package(OpenCV REQUIRED CONFIG)
include_directories(${OpenCV_INCLUDE_DIRS})

#-------------------------------------------------------------------------------
# Configure ResNet18 default benchmark.
#-------------------------------------------------------------------------------

set(RESNET18_TOSA_PIPELINE "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-tensor),func.func(tosa-to-linalg),func.func(tosa-to-arith))")

add_custom_command(OUTPUT Labels.txt ResNet-18-weighted.mlir
  COMMAND
    wget https://github.com/buddy-compiler/buddy-benchmark/raw/main/benchmarks/DeepLearning/Models/ResNet-18/Labels.txt
    wget https://github.com/buddy-compiler/buddy-benchmark/raw/main/benchmarks/DeepLearning/Models/ResNet-18/ResNet-18.mlir -O ResNet-18-weighted.mlir
  )

# Compile MLIR file to object file.
if(CMAKE_TOOLCHAIN_FILE STREQUAL ./riscv-toolchain.cmake)
  # Generate RISC-V object file.
  add_custom_command(OUTPUT resnet-18-default.o
  COMMAND
    ${LLVM_MLIR_BINARY_DIR}/mlir-opt ./ResNet-18-weighted.mlir
      --pass-pipeline="${RESNET18_TOSA_PIPELINE}" |
    ${LLVM_MLIR_BINARY_DIR}/mlir-opt
      --test-linalg-transform-patterns="test-generalize-pad-tensor"
      --linalg-bufferize
      --convert-linalg-to-loops
      --func-bufferize
      --arith-bufferize
      --tensor-bufferize
      --finalizing-bufferize
      --convert-vector-to-scf
      --convert-scf-to-cf
      --expand-strided-metadata
      --lower-affine
      --convert-vector-to-llvm
      --memref-expand
      --arith-expand
      --convert-arith-to-llvm
      --convert-memref-to-llvm
      --convert-math-to-llvm
      --llvm-request-c-wrappers
      --convert-func-to-llvm
      --reconcile-unrealized-casts |
    ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
    ${LLVM_MLIR_BINARY_DIR}/llc -O3 -mtriple riscv64 -target-abi lp64d
      -mattr=+m,+d,+v --filetype=obj -riscv-v-vector-bits-min=128
      -o ./resnet-18-default.o
  )
else()
  # Generate x86 object file.
  add_custom_command(OUTPUT resnet-18-default.o
  COMMAND
    ${LLVM_MLIR_BINARY_DIR}/mlir-opt ./ResNet-18-weighted.mlir
      --pass-pipeline="${RESNET18_TOSA_PIPELINE}" |
    ${LLVM_MLIR_BINARY_DIR}/mlir-opt
      --test-linalg-transform-patterns="test-generalize-pad-tensor"
      --linalg-bufferize
      --convert-linalg-to-loops
      --func-bufferize
      --arith-bufferize
      --tensor-bufferize
      --finalizing-bufferize
      --convert-vector-to-scf
      --convert-scf-to-cf
      --expand-strided-metadata
      --lower-affine
      --convert-vector-to-llvm
      --memref-expand
      --arith-expand
      --convert-arith-to-llvm
      --convert-memref-to-llvm
      --convert-math-to-llvm
      --llvm-request-c-wrappers
      --convert-func-to-llvm
      --reconcile-unrealized-casts |
    ${LLVM_MLIR_BINARY_DIR}/mlir-translate --mlir-to-llvmir |
    ${LLVM_MLIR_BINARY_DIR}/llc -mtriple=x86_64-unknown-linux-gnu -mattr=avx512f
      --filetype=obj -o ./resnet-18-default.o
  )
endif()

add_library(ResNet18Default STATIC resnet-18-default.o)
set_target_properties(ResNet18Default PROPERTIES LINKER_LANGUAGE CXX)

add_executable(resnet-18-benchmark ResNet18DefaultBenchmark.cpp)
# Link libraries
target_link_directories(resnet-18-benchmark PRIVATE ${LLVM_MLIR_LIBRARY_DIR})
target_link_libraries(resnet-18-benchmark
  ResNet18Default
  mlir_c_runner_utils
  ${OpenCV_LIBS}
)
