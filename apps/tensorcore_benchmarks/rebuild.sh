#!/bin/bash
set -euo pipefail

# Default values
BENCHMARK="conv1d"
TARGET="win"
SCHEDULE="cuda_only"
CONV_KERNEL_SIZE="128"
CONV_IMG_COL="3840"
CONV_IMG_ROW="2160"
MATMUL_M="4096"
MATMUL_N="4096"
MATMUL_K="4096"
VERIFY="false"

# Help message
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo "Build Halide benchmarks for different targets"
    echo ""
    echo "Options:"
    echo "  -b,         --benchmark NAME            Benchmark to build (conv1d, conv2d, or matmul) [default: conv1d]"
    echo "  -t,         --target TARGET             Target architecture (host, win, or linux) [default: host]"
    echo "  -s,         --schedule SCHEDULE         Schedule to use (cuda_only or tensorcore) [default: cuda_only]"
    echo "  -conv_k,    --kernel-size KERNEL_SIZE   Kernel size (128) [default: 128]"
    echo "  -conv_col,  --img-cols IMG_COL         Image width (3840) [default: 3840]"
    echo "  -conv_row,  --img-rows IMG_ROW         Image height (2160) [default: 2160]"
    echo "  -mm_m,      --matmul-m MATMUL_M         Rows of the input matrix A [default: 4096]"
    echo "  -mm_n,      --matmul-n MATMUL_N         Columns of the input matrix B [default: 4096]"
    echo "  -mm_k,      --matmul-k MATMUL_K         Columns of the input matrix A / Rows of the input matrix B [default: 4096]"
    echo "  -mm_mnk,    --matmul-dims DIM_SIZE      M,N,K Dimensions of the input matrix A and B [default: 4096]"
    echo "  -h, --help                              Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 -b conv1d -t win                  # Build conv1d for Windows"
    echo "  $0 -b conv2d -t linux                # Build conv2d for Linux"
    echo "  $0 -b conv2d -t linux -s tensorcore  # Build conv2d for Linux with tensorcore schedule"
    echo "  $0 -b matmul -t linux -mm_mnk 1024   # Build matmul for Linux with 1024x1024 matrices"
    echo "  $0                                   # Build conv1d for host (default)"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -b|--benchmark)
            BENCHMARK="$2"
            shift 2
            ;;
        -t|--target)
            TARGET="$2"
            shift 2
            ;;
        -s|--schedule)
            SCHEDULE="$2"
            shift 2
            ;;
        -conv_k|--kernel-size)
            CONV_KERNEL_SIZE="$2"
            shift 2
            ;;
        -conv_col|--img-cols)
            CONV_IMG_COL="$2"
            shift 2
            ;;
        -conv_row|--img-rows)
            CONV_IMG_ROW="$2"
            shift 2
            ;;
        -mm_m|--matmul-m)
            MATMUL_M="$2"
            shift 2
            ;;
        -mm_n|--matmul-n)
            MATMUL_N="$2"
            shift 2
            ;;
        -mm_k|--matmul-k)
            MATMUL_K="$2"
            shift 2
            ;;
        -mm_mnk|--matmul-dims)
            MATMUL_M="$2"
            MATMUL_N="$2"
            MATMUL_K="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        -v|--verify)
            shift 1
            ;;
        *)
            echo "Error: Unknown option $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate benchmark name
if [[ ! "$BENCHMARK" =~ ^(conv1d|conv2d|matmul|upsample|downsample|conv_layer)$ ]]; then
    echo "Error: Invalid benchmark name '$BENCHMARK'. Must be either 'conv1d', 'conv2d', 'upsample', 'downsample', 'matmul', or 'conv_layer'"
    exit 1
fi

# Validate target
if [[ ! "$TARGET" =~ ^(host|win|linux)$ ]]; then
    echo "Error: Invalid target '$TARGET'. Must be either 'host', 'win', or 'linux'"
    exit 1
fi

echo "Building $BENCHMARK for $TARGET target..."

# Clean and configure
rm -rf build
cmake -S . -B build \
    -DCMAKE_PREFIX_PATH=../../halide-install \
    -DSCHEDULE=$SCHEDULE \
    -DCONV_KERNEL_SIZE=$CONV_KERNEL_SIZE \
    -DCONV_IMG_COL=$CONV_IMG_COL \
    -DCONV_IMG_ROW=$CONV_IMG_ROW \
    -DMATMUL_M=$MATMUL_M \
    -DMATMUL_N=$MATMUL_N \
    -DMATMUL_K=$MATMUL_K \
    -DCMAKE_BUILD_TYPE=Release

# Build based on target
case "$TARGET" in
    win)
        cmake --build build --target ${BENCHMARK}_lib_win.update
        cmake --build build --target ${BENCHMARK}_lib_win.runtime.update
        ;;
    linux)
        cmake --build build --target ${BENCHMARK}_lib_linux.update
        cmake --build build --target ${BENCHMARK}_lib_linux.runtime.update
        ;;
    *)
        cmake --build build --target ${BENCHMARK}
        ;;
esac
