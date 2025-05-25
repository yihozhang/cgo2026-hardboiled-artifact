#!/bin/bash
set -euo pipefail

# Default values
BENCHMARK="conv1d"
TARGET="win"

# Help message
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo "Build Halide benchmarks for different targets"
    echo ""
    echo "Options:"
    echo "  -b, --benchmark NAME    Benchmark to build (conv1d or conv2d) [default: conv1d]"
    echo "  -t, --target TARGET     Target architecture (host, win, or linux) [default: host]"
    echo "  -h, --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 -b conv1d -t win     # Build conv1d for Windows"
    echo "  $0 -b conv2d -t linux   # Build conv2d for Linux"
    echo "  $0                      # Build conv1d for host (default)"
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
if [[ ! "$BENCHMARK" =~ ^(conv1d|conv2d)$ ]]; then
    echo "Error: Invalid benchmark name '$BENCHMARK'. Must be either 'conv1d' or 'conv2d'"
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
        cmake --build build --target ${BENCHMARK}_lib
        ;;
esac
