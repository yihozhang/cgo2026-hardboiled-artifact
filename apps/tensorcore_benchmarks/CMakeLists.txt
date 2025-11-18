cmake_minimum_required(VERSION 3.28)
project(tensorcore_benchmarks LANGUAGES CXX CUDA)

enable_testing()

# Set up language settings
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED YES)

# Conv parameters
set(CONV_KERNEL_SIZE 16 CACHE STRING "Size of the convolution kernel")
set(CONV_IMG_ROW 4096 CACHE STRING "Height of the input image")
set(CONV_IMG_COL 4096 CACHE STRING "Width of the input image")

# Resize parameters
set(UPSAMPLE true CACHE BOOL "Whether to upsample (true) or downsample (false)")

# NHWC parameters for NN layers
set(NN_TENSOR_N 128 CACHE STRING "Batch size (N) for NN layers")
set(NN_TENSOR_H 64 CACHE STRING "Image height (H) for NN layers")
set(NN_TENSOR_W 64 CACHE STRING "Image width (W) for NN layers")
set(NN_TENSOR_C 16 CACHE STRING "Channels (C) for NN layers")

# Attention parameters
set(ATT_D 128 CACHE STRING "Feature dimension (D) for attention")
set(ATT_L 2048 CACHE STRING "Sequence length (L) for attention")
set(ATT_N 8 CACHE STRING "Batch size (N) for attention")

# Matmul parameters
set(MATMUL_M 4096 CACHE STRING "Rows of the input matrix A")
set(MATMUL_N 4096 CACHE STRING "Columns of the input matrix B")
set(MATMUL_K 4096 CACHE STRING "Columns of the input matrix A / Rows of the input matrix B")

# General Parameters
set(SCHEDULE "tensorcore" CACHE STRING "GPU schedule to use (cudaonly or tensorcore)")
set(VERIFY_OUTPUT false CACHE STRING "Verify output was correct (true or false); might take long for large inputs")

# Find Halide
find_package(Halide REQUIRED)

# Find PNG and JPEG for image I/O
find_package(PNG REQUIRED)
find_package(JPEG REQUIRED)

# Find CUDA and cuBLAS for baseline comparisons
find_package(CUDAToolkit REQUIRED)

# Find cuDNN manually (not part of CUDAToolkit package)
find_library(CUDNN_LIBRARY cudnn HINTS ${CUDAToolkit_LIBRARY_DIR} PATH_SUFFIXES lib lib64)
find_path(CUDNN_INCLUDE_DIR cudnn.h HINTS ${CUDAToolkit_INCLUDE_DIRS} PATH_SUFFIXES include)
if(NOT CUDNN_LIBRARY)
    message(FATAL_ERROR "cuDNN library not found")
endif()
if(NOT CUDNN_INCLUDE_DIR)
    message(FATAL_ERROR "cuDNN headers not found")
endif()
message(STATUS "Found cuDNN: ${CUDNN_LIBRARY}")
message(STATUS "Found cuDNN headers: ${CUDNN_INCLUDE_DIR}")

# Find Pytorch
set(CMAKE_PREFIX_PATH "/opt/libtorch")
find_package(Torch REQUIRED)

# Include helper functions
include(cmake/TensorCoreHelpers.cmake)

# Add Halide Generators
add_halide_generator(conv1d.generator SOURCES conv1d_generator.cpp)
add_halide_generator(conv2d.generator SOURCES conv2d_generator.cpp)
add_halide_generator(upsample.generator SOURCES upsample_generator.cpp)
add_halide_generator(downsample.generator SOURCES downsample_generator.cpp)
add_halide_generator(denoise.generator SOURCES denoise_generator.cpp)
add_halide_generator(resize.generator SOURCES resize_generator.cpp)
add_halide_generator(rec_filter.generator SOURCES rec_filter_generator.cpp)
add_halide_generator(matmul.generator SOURCES matmul_generator.cpp)
add_halide_generator(conv_layer.generator SOURCES conv_layer_generator.cpp)
add_halide_generator(attention.generator SOURCES attention_generator.cpp)

# Create compiled libraries (adding extra targets for cross-compilation)
add_conv_library(conv1d_lib conv1d host-cuda-cuda_capability_80)
add_conv_library(conv1d_lib_win conv1d x86-64-windows-cuda-cuda_capability_80)
add_conv_library(conv1d_lib_linux conv1d x86-64-linux-cuda-cuda_capability_80)

add_conv_library(conv2d_lib conv2d host-cuda-cuda_capability_80)
add_conv_library(conv2d_lib_win conv2d x86-64-windows-cuda-cuda_capability_80)
add_conv_library(conv2d_lib_linux conv2d x86-64-linux-cuda-cuda_capability_80)

add_conv_library(upsample_lib upsample host-cuda-cuda_capability_80)
add_conv_library(upsample_lib_win upsample x86-64-windows-cuda-cuda_capability_80)
add_conv_library(upsample_lib_linux upsample x86-64-linux-cuda-cuda_capability_80)

add_conv_library(downsample_lib downsample host-cuda-cuda_capability_80)
add_conv_library(downsample_lib_win downsample x86-64-windows-cuda-cuda_capability_80)
add_conv_library(downsample_lib_linux downsample x86-64-linux-cuda-cuda_capability_80)

add_denoise_library(denoise_lib denoise host-cuda-cuda_capability_80)
add_denoise_library(denoise_lib_win denoise x86-64-windows-cuda-cuda_capability_80)
add_denoise_library(denoise_lib_linux denoise x86-64-linux-cuda-cuda_capability_80)

add_resize_library(resize resize host-cuda-cuda_capability_80 "")
add_resize_library(resize resize x86-64-windows-cuda-cuda_capability_80 "_win")
add_resize_library(resize resize x86-64-linux-cuda-cuda_capability_80 "_linux")

add_recfilter_library(rec_filter_lib rec_filter host-cuda-cuda_capability_80)
add_recfilter_library(rec_filter_lib_win rec_filter x86-64-windows-cuda-cuda_capability_80)
add_recfilter_library(rec_filter_lib_linux rec_filter x86-64-linux-cuda-cuda_capability_80)

add_matmul_library(matmul_lib matmul host-cuda-cuda_capability_80)
add_matmul_library(matmul_lib_win matmul x86-64-windows-cuda-cuda_capability_80)
add_matmul_library(matmul_lib_linux matmul x86-64-linux-cuda-cuda_capability_80)

add_nnl_library(conv_layer_lib conv_layer host-cuda-cuda_capability_80)
add_nnl_library(conv_layer_lib_win conv_layer x86-64-windows-cuda-cuda_capability_80)
add_nnl_library(conv_layer_lib_linux conv_layer x86-64-linux-cuda-cuda_capability_80)

add_attn_library(attention_lib attention host-cuda-cuda_capability_80)
add_attn_library(attention_lib_win attention x86-64-windows-cuda-cuda_capability_80)
add_attn_library(attention_lib_linux attention x86-64-linux-cuda-cuda_capability_80)

# Create executables (only for host)
add_executable(conv1d process.cpp)
add_executable(conv2d process.cpp)
add_executable(matmul process.cpp)
add_executable(upsample process.cpp)
add_executable(downsample process.cpp)
add_executable(denoise process.cpp)
add_executable(resize process.cpp)
add_executable(rec_filter process.cpp)
add_executable(conv_layer process.cpp)
add_executable(attention process.cpp)

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

target_compile_definitions(upsample
    PRIVATE
    BENCHMARK_HEADER="upsample_lib.h"
    BENCHMARK_NAME="upsample"
    RUN_upsample
    KERNEL_SIZE=${CONV_KERNEL_SIZE}
    IMG_ROW=${CONV_IMG_ROW}
    IMG_COL=${CONV_IMG_COL}
    SCHEDULE="${SCHEDULE}"
    VERIFY_OUTPUT=${VERIFY_OUTPUT}
)

target_compile_definitions(downsample
    PRIVATE
    BENCHMARK_HEADER="downsample_lib.h"
    BENCHMARK_NAME="downsample"
    RUN_downsample
    KERNEL_SIZE=${CONV_KERNEL_SIZE}
    IMG_ROW=${CONV_IMG_ROW}
    IMG_COL=${CONV_IMG_COL}
    SCHEDULE="${SCHEDULE}"
    VERIFY_OUTPUT=${VERIFY_OUTPUT}
)

target_compile_definitions(denoise
    PRIVATE
    BENCHMARK_HEADER="denoise_lib.h"
    BENCHMARK_NAME="denoise"
    RUN_denoise
    IMG_ROW=${CONV_IMG_ROW}
    IMG_COL=${CONV_IMG_COL}
    SCHEDULE="${SCHEDULE}"
    VERIFY_OUTPUT=${VERIFY_OUTPUT}
)

target_compile_definitions(resize
    PRIVATE
    BENCHMARK_HEADER="resize_up_lib.h"
    BENCHMARK_HEADER_EXTRA="resize_down_lib.h"
    BENCHMARK_NAME="resize"
    RUN_resize
    IMG_ROW=${CONV_IMG_ROW}
    IMG_COL=${CONV_IMG_COL}
    SCHEDULE="${SCHEDULE}"
    VERIFY_OUTPUT=${VERIFY_OUTPUT}
)

target_compile_definitions(rec_filter
    PRIVATE
    BENCHMARK_HEADER="rec_filter_lib.h"
    BENCHMARK_NAME="rec_filter"
    RUN_rec_filter
    IMG_COL=${CONV_IMG_COL}
    SCHEDULE="${SCHEDULE}"
    VERIFY_OUTPUT=${VERIFY_OUTPUT}
)

target_compile_definitions(matmul
    PRIVATE
    BENCHMARK_HEADER="matmul_lib.h"
    BENCHMARK_NAME="matmul"
    RUN_matmul
    MATMUL_M=${MATMUL_M}
    MATMUL_N=${MATMUL_N}
    MATMUL_K=${MATMUL_K}
    SCHEDULE="${SCHEDULE}"
    VERIFY_OUTPUT=${VERIFY_OUTPUT}
)

target_compile_definitions(conv_layer
    PRIVATE
    BENCHMARK_HEADER="conv_layer_lib.h"
    BENCHMARK_NAME="conv_layer"
    RUN_conv_layer
    KERNEL_SIZE=${CONV_KERNEL_SIZE}
    NN_TENSOR_N=${NN_TENSOR_N}
    NN_TENSOR_H=${NN_TENSOR_H}
    NN_TENSOR_W=${NN_TENSOR_W}
    NN_TENSOR_C=${NN_TENSOR_C}
    SCHEDULE="${SCHEDULE}"
    VERIFY_OUTPUT=${VERIFY_OUTPUT}
)

target_compile_definitions(attention
    PRIVATE
    BENCHMARK_HEADER="attention_lib.h"
    BENCHMARK_NAME="attention"
    RUN_attention
    ATT_D=${ATT_D}
    ATT_L=${ATT_L}
    ATT_N=${ATT_N}
    SCHEDULE="${SCHEDULE}"
    VERIFY_OUTPUT=${VERIFY_OUTPUT}
)

target_link_libraries(conv1d
    PRIVATE
    conv1d_lib
    Halide::Halide
    Halide::Tools
)

target_link_libraries(conv2d
    PRIVATE
    conv2d_lib
    Halide::Halide
    Halide::Tools
)

target_link_libraries(upsample
    PRIVATE
    upsample_lib
    Halide::Halide
    Halide::Tools
)

target_link_libraries(downsample
    PRIVATE
    downsample_lib
    Halide::Halide
    Halide::Tools
)

target_link_libraries(denoise
    PRIVATE
    denoise_lib
    Halide::Halide
    Halide::Tools
    PNG::PNG
    JPEG::JPEG
)

target_link_libraries(resize
    PRIVATE
    resize_up_lib
    resize_down_lib
    Halide::Halide
    Halide::Tools
    PNG::PNG
    JPEG::JPEG
)

target_link_libraries(rec_filter
    PRIVATE
    rec_filter_lib
    Halide::Halide
    Halide::Tools
)

target_link_libraries(matmul
    PRIVATE
    matmul_lib
    Halide::Halide
    Halide::Tools
)

target_link_libraries(conv_layer
    PRIVATE
    conv_layer_lib
    Halide::Halide
    Halide::Tools
)

target_link_libraries(attention
    PRIVATE
    attention_lib
    Halide::Halide
    Halide::Tools
)

# Baselines 
add_executable(matmul_cublas baselines/matmul_baseline.cpp)
add_executable(conv_layer_pytorch baselines/conv_layer_pytorch.cpp)
add_executable(conv_layer_cudnn baselines/conv_layer_cudnn.cpp)
add_executable(attention_pytorch baselines/attention_pytorch.cpp)
add_executable(attention_cudnn baselines/attention_cudnn.cu)

target_link_libraries(matmul_cublas
    PRIVATE
    Halide::Tools
    CUDA::cudart
    CUDA::cublas
    CUDA::cublasLt
)

target_link_libraries(conv_layer_pytorch
    PRIVATE
    "${TORCH_LIBRARIES}"
    Halide::Tools
)

target_link_libraries(conv_layer_cudnn
    PRIVATE
    Halide::Tools
    CUDA::cudart
    ${CUDNN_LIBRARY}
)

target_link_libraries(attention_pytorch
    PRIVATE
    "${TORCH_LIBRARIES}"
    Halide::Tools
)

target_link_libraries(attention_cudnn
    PRIVATE
    Halide::Tools
    CUDA::cudart
    CUDA::cublas
    CUDA::cublasLt
    ${CUDNN_LIBRARY}
)

target_compile_definitions(matmul_cublas
    PRIVATE
    MATMUL_M=${MATMUL_M}
    MATMUL_N=${MATMUL_N}
    MATMUL_K=${MATMUL_K}
    VERIFY_OUTPUT=${VERIFY_OUTPUT}
)

target_compile_definitions(conv_layer_cudnn
    PRIVATE
    NN_TENSOR_N=${NN_TENSOR_N}
    NN_TENSOR_H=${NN_TENSOR_H}
    NN_TENSOR_W=${NN_TENSOR_W}
    NN_TENSOR_C=${NN_TENSOR_C}
    KERNEL_SIZE=${CONV_KERNEL_SIZE}
)

target_compile_definitions(conv_layer_pytorch 
    PRIVATE
    _GLIBCXX_USE_CXX11_ABI=1    
    PRIVATE
    NN_TENSOR_N=${NN_TENSOR_N}
    NN_TENSOR_H=${NN_TENSOR_H}
    NN_TENSOR_W=${NN_TENSOR_W}
    NN_TENSOR_C=${NN_TENSOR_C}
    KERNEL_SIZE=${CONV_KERNEL_SIZE}
)

target_compile_definitions(attention_cudnn
    PRIVATE
    ATT_D=${ATT_D}
    ATT_L=${ATT_L}
    ATT_N=${ATT_N}
)

target_compile_definitions(attention_pytorch
    PRIVATE
    ATT_D=${ATT_D}
    ATT_L=${ATT_L}
    ATT_N=${ATT_N}
    _GLIBCXX_USE_CXX11_ABI=1
)

target_include_directories(attention_cudnn PRIVATE ${CUDNN_INCLUDE_DIR})
target_include_directories(conv_layer_cudnn PRIVATE ${CUDNN_INCLUDE_DIR})


