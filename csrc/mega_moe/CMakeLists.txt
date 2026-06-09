# =============================================================================
# quack/csrc/mega_moe — TVM-FFI shared lib exposing TE's BF16 MegaMoE megakernel
# -----------------------------------------------------------------------------
# Produces libquack_mega_moe.so (TVM-FFI ABI), loaded by quack/mega_moe.py via
# tvm_ffi.load_module. Mirrors /Users/min.yang/github/mega_moe/CMakeLists.txt and
# the build flags in <TE>/qa/megamoe_bf16_launch.cu.
#
# REQUIRES an SM100 (Blackwell) toolchain — CANNOT be built/tested on non-CUDA hosts.
#
# The MegaMoE kernel + host launcher are VENDORED flat in this directory
# (mega_moe_launch.cuh / mega_moe_config.h + impls/ comm/ common/ layout/ mma/ ptx/ scheduler/),
# so the build is self-contained — only CUTLASS and tvm-ffi are external.
#
# Configure (B200 box):
#   cmake -B build csrc/mega_moe \
#     -DQUACK_CUTLASS_DIR=/path/to/cutlass \
#     -DQUACK_TVM_FFI_DIR=$(python -c "import tvm_ffi,os;print(os.path.dirname(tvm_ffi.__file__))")
#   cmake --build build -j
#   export QUACK_MEGA_MOE_LIB=$PWD/build/libquack_mega_moe.so
# =============================================================================
cmake_minimum_required(VERSION 3.24)
project(quack_mega_moe LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 100a)   # Blackwell sm_100a

set(QUACK_CUTLASS_DIR "" CACHE PATH "CUTLASS root (uses include/ and tools/util/include)")
set(QUACK_TVM_FFI_DIR "" CACHE PATH "apache-tvm-ffi package dir (has include/ and lib/)")

foreach(v QUACK_CUTLASS_DIR QUACK_TVM_FFI_DIR)
  if(NOT ${v})
    message(FATAL_ERROR "${v} must be set (-D${v}=...)")
  endif()
endforeach()

add_library(quack_mega_moe SHARED mega_moe_ffi.cu)
set_target_properties(quack_mega_moe PROPERTIES PREFIX "lib")

target_include_directories(quack_mega_moe PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}                 # flat root: <mega_moe_launch.cuh> <impls/..> <common/..> ...
    ${QUACK_CUTLASS_DIR}/include
    ${QUACK_CUTLASS_DIR}/tools/util/include
    ${QUACK_TVM_FFI_DIR}/include)

target_link_directories(quack_mega_moe PRIVATE ${QUACK_TVM_FFI_DIR}/lib)
target_link_libraries(quack_mega_moe PRIVATE tvm_ffi cuda)

target_compile_options(quack_mega_moe PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --expt-extended-lambda -O3>)
