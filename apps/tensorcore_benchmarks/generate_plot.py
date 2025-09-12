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
    plt.xlabel('Kernel Size', fontsize=25)
    plt.ylabel('Execution Time (ms)', fontsize=25)
    plt.title(f'{title} Performance Comparison', fontsize=28, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Set x-axis to show all kernel sizes
    plt.xticks(kernel_sizes[::3], fontsize=20)  # Show every other size to avoid crowding
    # Make y-axis start from 0 for better comparison
    # plt.ylim(bottom=0, top=)
    ymin, ymax = 0, max(np.max(cudaonly_times), np.max(tensorcore_times)) * 1.05
    plt.yticks(list(range(ymin, int(np.ceil(ymax)), 1)), fontsize=20)
    plt.legend(fontsize=20)
    
    # Add some styling
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def do_benchmark_line_plot(benchmark_name, title, kernel_sizes):
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
                              save_path=f'{benchmark_name}_performance_comparison.pdf')

def plot_benchmark_bar_chart_single_kernel(data, benchmarks, kernel_sizes, idx, save_path=None):
    """
    Create a grouped bar chart comparing benchmark performance across different
    benchmarks and kernel sizes.
    """
    import numpy as np
    
    # Create labels for x-axis (e.g., "Conv2D-16", "Conv2D-32", etc.)
    labels = []
    cudaonly_values = []
    tensorcore_values = []
    
    for benchmark in benchmarks:
        i = idx
        kernel_size = kernel_sizes[idx]
        labels.append(f'{benchmark.capitalize()}')
        
        cuda_time = data[benchmark]['cudaonly'][i]
        tensor_time = data[benchmark]['tensorcore'][i]
        
        # Use 0 for None values, but we'll handle this in plotting
        cudaonly_values.append(cuda_time if cuda_time is not None else 0)
        tensorcore_values.append(tensor_time if tensor_time is not None else 0)
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create bars
    bars1 = ax.bar(x - width/2, cudaonly_values, width, 
                  label='Halide (CUDA)', color='#1f77b4', alpha=0.8)
    bars2 = ax.bar(x + width/2, tensorcore_values, width,
                  label='Halide (Tensor Cores)', color='#ff7f0e', alpha=0.8)
    
    # Customize the plot
    ax.set_xlabel(f'Benchmark', fontsize=25)
    ax.set_ylabel('Execution Time (ms)', fontsize=25)
    ax.set_title(f'Performance Comparison on 2D microbenchmarks (k={kernel_sizes[idx]})', fontsize=28, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=20)
    ax.tick_params(axis="y", labelsize=20)
    ax.legend(fontsize=20)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Set y-axis to start from 0
    ax.set_ylim(top=max(np.max(cudaonly_values),np.max(tensorcore_values))*1.2, bottom=0)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        if height > 0:  # Only label non-zero values
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=18)
    
    for bar in bars2:
        height = bar.get_height()
        if height > 0:  # Only label non-zero values
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=18)
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Bar chart saved to {save_path}")
    else:
        plt.show()

def do_benchmark_bar_chart():
    """
    Create two bar charts comparing performance across conv2d, upsample, and downsample
    - one for kernel size 16 and one for kernel size 32.
    """
    benchmarks = ['conv2d', 'downsample', 'upsample']
    kernel_sizes = [16, 32]
    
    print("Collecting benchmark data for bar charts...")
    
    # Collect data for all benchmarks
    data = {}
    for benchmark in benchmarks:
        print(f"Running {benchmark} benchmarks...")
        cudaonly_times, tensorcore_times = collect_benchmark_data(benchmark, kernel_sizes)
        data[benchmark] = {
            'cudaonly': cudaonly_times,
            'tensorcore': tensorcore_times
        }
    
    print("Data collection complete. Creating bar charts...")
    
    # Filter out benchmarks with no valid data
    valid_benchmarks = []
    for benchmark in benchmarks:
        if any(t is not None for t in data[benchmark]['cudaonly']) and \
           any(t is not None for t in data[benchmark]['tensorcore']):
            valid_benchmarks.append(benchmark)
    
    if not valid_benchmarks:
        print("No valid benchmark data collected for any benchmark.")
        return
    
    # Create bar chart for kernel size 16 (index 0)
    plot_benchmark_bar_chart_single_kernel(data, valid_benchmarks, kernel_sizes, 0, 
                                          save_path='benchmark_comparison_kernel_16.pdf')
    
    # Create bar chart for kernel size 32 (index 1)
    plot_benchmark_bar_chart_single_kernel(data, valid_benchmarks, kernel_sizes, 1, 
                                          save_path='benchmark_comparison_kernel_32.pdf')
if __name__ == "__main__":
    # do_benchmark_line_plot("conv1d", "Conv1D", range(8, 129, 8))
    # do_benchmark_line_plot("conv1d", "Conv1D", range(8, 33, 8))
    do_benchmark_bar_chart()