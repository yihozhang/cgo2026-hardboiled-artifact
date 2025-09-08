#!/bin/bash
set -euo pipefail
set -x

# Default values
BENCHMARK="conv1d"
TARGET="win"
SCHEDULE="cuda_only"
CONV_KERNEL_SIZE="128"
CONV_IMG_COL="3840"
CONV_IMG_ROW="2160"
NN_TENSOR_N="128"
NN_TENSOR_H="64"
NN_TENSOR_W="64"
NN_TENSOR_C="32"
ATT_D="128"
ATT_L="2048"
ATT_N="8"
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
    echo "  -b,         --benchmark NAME            Benchmark to build (conv1d, conv2d, upsample, downsample, matmul, conv_layer, attention) [default: conv1d]"
    echo "  -t,         --target TARGET             Target architecture (host, win, or linux) [default: host]"
    echo "  -s,         --schedule SCHEDULE         Schedule to use (cuda_only or tensorcore) [default: cuda_only]"
    echo "  -conv_k,    --kernel-size KERNEL_SIZE   Kernel size (128) [default: 128]"
    echo "  -conv_col,  --img-cols IMG_COL          Image width (3840) [default: 3840]"
    echo "  -conv_row,  --img-rows IMG_ROW          Image height (2160) [default: 2160]"
    echo "  -nhwc,      --nhwc N H W C              Batch size N, tensor height H, tensor width W, and tensor channels C [default: 128 64 64 32]"
    echo "  -att,       --att D L N                 Attention dims D, L, N [default: 128 2048 8]"
    echo "  -mm_mnk,    --matmul-dims M N K         M, N, K Dimensions of the input matrix A and B [default: 4096 4096 4096]"
    echo "  -h, --help                              Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 -b conv1d -t win                     # Build conv1d for Windows"
    echo "  $0 -b conv2d -t linux                   # Build conv2d for Linux"
    echo "  $0 -b conv2d -t linux -s tensorcore     # Build conv2d for Linux with tensorcore schedule"
    echo "  $0 -b matmul -t linux -mm_mnk 1024      # Build matmul for Linux with 1024x1024 matrices"
    echo "  $0 -b attention -t host -att 128 64 8   # Build attention for host with D=128 L=64 N=8"
    echo "  $0                                      # Build conv1d for host (default)"
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
        -nhwc|--nhwc)
            NN_TENSOR_N="$2"
            NN_TENSOR_H="$3"
            NN_TENSOR_W="$4"
            NN_TENSOR_C="$5"
            shift 5
            ;;
        -att|--att)
            ATT_D="$2"
            ATT_L="$3"
            ATT_N="$4"
            shift 4
            ;;
        -mm_mnk|--matmul-dims)
            MATMUL_M="$2"
            MATMUL_N="$3"
            MATMUL_K="$4"
            shift 4
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
if [[ ! "$BENCHMARK" =~ ^(conv1d|conv2d|matmul|upsample|downsample|conv_layer|attention)$ ]]; then
    echo "Error: Invalid benchmark name '$BENCHMARK'. Must be one of: conv1d, conv2d, upsample, downsample, matmul, conv_layer, attention"
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
    -DNN_TENSOR_N=$NN_TENSOR_N \
    -DNN_TENSOR_H=$NN_TENSOR_H \
    -DNN_TENSOR_W=$NN_TENSOR_W \
    -DNN_TENSOR_C=$NN_TENSOR_C \
    -DATT_D=$ATT_D \
    -DATT_L=$ATT_L \
    -DATT_N=$ATT_N \
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
