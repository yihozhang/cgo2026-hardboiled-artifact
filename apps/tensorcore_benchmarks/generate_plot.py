#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import subprocess
import os
import re

def run_benchmark(benchmark_name, schedule, kernel_size, bin_path="./bin"):
    """
    Run the benchmark for a given schedule and kernel size.
    Returns the execution time in milliseconds.
    """
    target_path = f"{bin_path}/{schedule}_{kernel_size}/{benchmark_name}"
    
    os.system(f"make {target_path}")
    
    try:
        # Run the benchmark and capture output
        result = subprocess.run([target_path], capture_output=True, text=True, timeout=1000)
        output = result.stdout
        
        time_match = re.search(r'Runtime:\s*([0-9.]+)\n', output)
        if time_match:
            return float(time_match.group(1)) * 1000
        else:
            print(f"Could not parse timing from output: {output}")
            return None
            
    except subprocess.TimeoutExpired:
        print(f"Benchmark {target_path} timed out")
        return None
    except Exception as e:
        print(f"Error running {target_path}: {e}")
        return None

def collect_benchmark_data(benchmark_name, kernel_sizes, bin_path="./bin"):
    """
    Collect actual benchmark data by running the compiled binaries.
    """
    schedules = ['cudaonly', 'tensorcore']
    
    data = {}
    
    for schedule in schedules:
        times = []
        for kernel_size in kernel_sizes:
            print(f"Running {schedule} with kernel size {kernel_size}...")
            time = run_benchmark(benchmark_name, schedule, kernel_size, bin_path)
            times.append(time)
        data[schedule] = times
    
    return data['cudaonly'], data['tensorcore']

def plot_performance_comparison(title, kernel_sizes, cudaonly_times, tensorcore_times, save_path=None):
    """
    Create a line plot comparing the performance of two schedules.
    """
    plt.figure(figsize=(12, 8))
    
    # Plot lines
    plt.plot(kernel_sizes, cudaonly_times, 'o-', linewidth=2, markersize=6, 
             label='Halide (CUDA)', color='#1f77b4', alpha=0.8)
    plt.plot(kernel_sizes, tensorcore_times, 's-', linewidth=2, markersize=6, 
             label='Halide (Tensor Cores)', color='#ff7f0e', alpha=0.8)
    
    # Customize the plot
    plt.xlabel('Kernel Size', fontsize=12)
    plt.ylabel('Execution Time (ms)', fontsize=12)
    plt.title(f'{title} Performance Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Set x-axis to show all kernel sizes
    plt.xticks(kernel_sizes[::1], rotation=45)  # Show every other size to avoid crowding
    
    # Make y-axis start from 0 for better comparison
    plt.ylim(bottom=0)
    
    # Add some styling
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def do_benchmark(benchmark_name, title, kernel_sizes):
    print("Collecting benchmark data...")
    kernel_sizes = list(kernel_sizes)
    cudaonly_times, tensorcore_times = collect_benchmark_data(benchmark_name, kernel_sizes)
    
    # Filter out None values (failed benchmarks)
    valid_data = [(k, c, t) for k, c, t in zip(kernel_sizes, cudaonly_times, tensorcore_times) 
                    if c is not None and t is not None]
    print(valid_data)
    if not valid_data:
        print("No valid benchmark data collected.")
    else:
        kernel_sizes, cudaonly_times, tensorcore_times = zip(*valid_data)
    
    # Create the plot
    plot_performance_comparison(title, kernel_sizes, cudaonly_times, tensorcore_times, 
                              save_path=f'{benchmark_name}_performance_comparison.png')

if __name__ == "__main__":
    do_benchmark("conv1d", "Conv1D", range(8, 129, 8))
    do_benchmark("conv2d", "Conv2D", range(8, 33, 8))
    do_benchmark("upsample", "Upsample", range(16, 33, 16)) 
    do_benchmark("downsample", "Downsample", range(16, 33, 16)) 