cmake_minimum_required(VERSION 3.28)
project(tensorcore_benchmarks)

enable_testing()

# Set up language settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

# Conv parameters
set(CONV_KERNEL_SIZE 128 CACHE STRING "Size of the convolution kernel")
set(CONV_IMG_ROW 61440 CACHE STRING "Height of the input image")
set(CONV_IMG_COL 34560 CACHE STRING "Width of the input image")

# General Parameters
set(SCHEDULE "tensorcore" CACHE STRING "GPU schedule to use (cuda_only or tensorcore)")
set(VERIFY_OUTPUT false CACHE STRING "Verify output was correct (true or false); might take long for large inputs")

# Find Halide
find_package(Halide REQUIRED)

# Include helper functions
include(cmake/TensorCoreHelpers.cmake)

# Add Halide Generators
add_halide_generator(conv1d.generator SOURCES conv1d_generator.cpp)
add_halide_generator(conv2d.generator SOURCES conv2d_generator.cpp)

# Create compiled libraries (adding extra targets for cross-compilation)
add_conv_library(conv1d_lib conv1d host-cuda-cuda_capability_86)
add_conv_library(conv1d_lib_win conv1d x86-64-windows-cuda-cuda_capability_86-no_asserts-no_bounds_query)
add_conv_library(conv1d_lib_linux conv1d x86-64-linux-cuda-cuda_capability_86-no_asserts-no_bounds_query)

add_conv_library(conv2d_lib conv2d host-cuda-cuda_capability_86)
add_conv_library(conv2d_lib_win conv2d x86-64-windows-cuda-cuda_capability_86-no_asserts-no_bounds_query)
add_conv_library(conv2d_lib_linux conv2d x86-64-linux-cuda-cuda_capability_86-no_asserts-no_bounds_query)

# Create executables (only for host)
add_executable(conv1d process.cpp)
add_executable(conv2d process.cpp)

target_compile_definitions(conv1d
    PRIVATE
    BENCHMARK_HEADER="conv1d_lib.h"
    BENCHMARK_NAME="conv1d"
    RUN_conv1d
    KERNEL_SIZE=${CONV_KERNEL_SIZE}
    IMG_ROW=${CONV_IMG_ROW}
    IMG_COL=${CONV_IMG_COL}
    SCHEDULE="${SCHEDULE}"
    VERIFY_OUTPUT=${VERIFY_OUTPUT}
)
target_compile_definitions(conv2d
    PRIVATE
    BENCHMARK_HEADER="conv2d_lib.h"
    BENCHMARK_NAME="conv2d"
    RUN_conv2d
    KERNEL_SIZE=${CONV_KERNEL_SIZE}
    IMG_ROW=${CONV_IMG_ROW}
    IMG_COL=${CONV_IMG_COL}
    SCHEDULE="${SCHEDULE}"
    VERIFY_OUTPUT=${VERIFY_OUTPUT}
)

target_link_libraries(conv1d
    PRIVATE
    conv1d_lib
    Halide::Tools
)

target_link_libraries(conv2d
    PRIVATE
    conv2d_lib
    Halide::Tools
)