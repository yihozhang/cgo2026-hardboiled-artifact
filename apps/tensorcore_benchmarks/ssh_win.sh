#!/bin/bash
set -euo pipefail

########################################
# CONFIG:
########################################
SSH_USER="maazs"
SSH_HOST="192.168.4.123"

########################################

# Default values
BENCHMARK="conv1d"
SCHEDULE="cuda_only"
CONV_KERNEL_SIZE="128"
CONV_IMG_COLS="3840"
CONV_IMG_ROWS="2160"
MATMUL_M="4096"
MATMUL_N="4096"
MATMUL_K="4096"
VERIFY="false"

# Help message
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo "Run benchmarks on Windows remote machine"
    echo ""
    echo "Options:"
    echo "  -b,         --benchmark NAME            Benchmark to run (conv1d, conv2d, or matmul) [default: conv1d]"
    echo "  -s,         --schedule SCHEDULE         Schedule to use (cuda_only or tensorcore) [default: cuda_only]"
    echo "  -conv_k,    --kernel-size KERNEL_SIZE   Kernel size for conv [default: 128]"
    echo "  -conv_col,  --img-cols COLS            Image columns for conv [default: 2160]"
    echo "  -conv_row,  --img-rows ROWS            Image rows for conv [default: 3840]"
    echo "  -mm_m,      --matmul-m MATMUL_M        Rows of the input matrix A [default: 4096]"
    echo "  -mm_n,      --matmul-n MATMUL_N        Columns of the input matrix B [default: 4096]"
    echo "  -mm_k,      --matmul-k MATMUL_K        Columns of the input matrix A / Rows of the input matrix B [default: 4096]"
    echo "  -mm_mnk,    --matmul-dims DIM_SIZE     M,N,K Dimensions of the input matrix A and B [default: 4096]"
    echo "  -v,         --verify                   Verify output [default: false]"
    echo "  -h,         --help                     Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 -b conv1d -s tensorcore -conv_k 128 -conv_col 2160 -conv_row 3840"
    echo "  $0 -b matmul -s tensorcore -mm_mnk 1024"
    echo "  $0 -b conv2d -s cuda_only -conv_k 128 -v"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -b|--benchmark)
            BENCHMARK="$2"
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
            CONV_IMG_COLS="$2"
            shift 2
            ;;
        -conv_row|--img-rows)
            CONV_IMG_ROWS="$2"
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
        -v|--verify)
            VERIFY="true"
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Error: Unknown option $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate benchmark name
if [[ ! "$BENCHMARK" =~ ^(conv1d|conv2d|matmul|upsample|downsample)$ ]]; then
    echo "Error: Invalid benchmark name '$BENCHMARK'. Must be either 'conv1d', 'conv2d', 'upsample', 'downsample', or 'matmul'"
    exit 1
fi

APP_DIR="/Users/mahmad/repos/public/YhHalide/apps/tensorcore_benchmarks"
HEADER_DIR="/Users/mahmad/repos/public/YhHalide/halide-install/include"
TOOL_DIR="/Users/mahmad/repos/public/YhHalide/tools/"
REMOTE_DIR='C:\Users\maazs\Desktop'

# 1) Copy the app directory
echo "⏳ Copying '$APP_DIR' → $SSH_USER@$SSH_HOST:$REMOTE_DIR …"
sshpass -p "$PASSWORD" scp -r -o StrictHostKeyChecking=no \
  "$APP_DIR" "$SSH_USER@$SSH_HOST:$REMOTE_DIR"

# 2) Copy the header directory
echo "⏳ Copying '$HEADER_DIR' → $SSH_USER@$SSH_HOST:$REMOTE_DIR\tensorcore_benchmarks …"
sshpass -p "$PASSWORD" scp -r -o StrictHostKeyChecking=no \
  "$HEADER_DIR" "$SSH_USER@$SSH_HOST:$REMOTE_DIR\tensorcore_benchmarks"

# 3) Copy the tool directory
echo "⏳ Copying '$TOOL_DIR' → $SSH_USER@$SSH_HOST:$REMOTE_DIR\tensorcore_benchmarks …"
sshpass -p "$PASSWORD" scp -r -o StrictHostKeyChecking=no \
  "$TOOL_DIR" "$SSH_USER@$SSH_HOST:$REMOTE_DIR\tensorcore_benchmarks"

# Build command based on benchmark type
if [[ "$BENCHMARK" == "matmul" ]]; then
    BUILD_ARGS="$BENCHMARK $SCHEDULE $MATMUL_M $MATMUL_N $MATMUL_K $VERIFY"
else
    BUILD_ARGS="$BENCHMARK $SCHEDULE $CONV_KERNEL_SIZE $CONV_IMG_COLS $CONV_IMG_ROWS $VERIFY"
fi

# 4) SSH in and launch PowerShell at that path
echo "⏳ Running: .\\build.bat $BUILD_ARGS"
sshpass -p "$PASSWORD" ssh -T -o StrictHostKeyChecking=no \
  "$SSH_USER@$SSH_HOST" \
  "cmd.exe /c \"call \"C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Auxiliary\\Build\\vcvars64.bat\" && cd /d $REMOTE_DIR\\tensorcore_benchmarks\ && .\\build.bat $BUILD_ARGS && .\\process.exe\""

#sshpass -p "$PASSWORD" ssh -T -o StrictHostKeyChecking=no \
#  "$SSH_USER@$SSH_HOST" \
#  "cmd.exe /k \"call \"C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Auxiliary\\Build\\vcvars64.bat\" && cd /d $REMOTE_DIR\\tensorcore_benchmarks\""