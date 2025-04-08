#!/bin/bash

# Check if an argument is provided
if [ $# -eq 0 ]; then
    echo "Error: No target specified."
    echo "Usage: $0 <target_name>"
    exit 1
fi

# Get the target name from the first argument
TARGET=$1

# Execute the converted command
echo "Building target: instrsel-benchmarks_$TARGET"
cmake --build build --target instrsel-benchmarks_$TARGET && \
HL_DEBUG_CODEGEN=2 build/instrsel-benchmarks/instrsel-benchmarks_$TARGET
