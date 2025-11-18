#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import subprocess
import os
import re
import csv
import shutil
import argparse
import time
from pathlib import Path

def run_benchmark(benchmark_name, schedule, kernel_size, bin_path="./build"):
    """
    Run the benchmark for a given schedule and kernel size.
    Returns the execution time in milliseconds.
    """
    target_path = f"{bin_path}/{benchmark_name}"

    os.system(f"bash ./rebuild.sh -b {benchmark_name} -conv_k {kernel_size} -conv_col 4096 -conv_row 4096 -v -t host -s {schedule}")
    
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

def save_benchmark_results_to_csv(benchmark_name, kernel_sizes, cudaonly_times, tensorcore_times, cache_dir="./cache"):
    """
    Save benchmark results to a CSV file.
    """
    # Create cache directory if it doesn't exist
    Path(cache_dir).mkdir(exist_ok=True)
    
    csv_path = f"{cache_dir}/{benchmark_name}_results.csv"
    
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(['benchmark_name', 'schedule', 'kernel_size', 'execution_time_ms'])
        
        # Write CUDA results
        for kernel_size, time in zip(kernel_sizes, cudaonly_times):
            writer.writerow([benchmark_name, 'cudaonly', kernel_size, time])
        
        # Write TensorCore results
        for kernel_size, time in zip(kernel_sizes, tensorcore_times):
            writer.writerow([benchmark_name, 'tensorcore', kernel_size, time])
    
    print(f"Results saved to {csv_path}")

def load_benchmark_results_from_csv(benchmark_name, kernel_sizes, cache_dir="./cache"):
    """
    Load benchmark results from a CSV file.
    Returns cudaonly_times, tensorcore_times lists.
    """
    csv_path = f"{cache_dir}/{benchmark_name}_results.csv"
    
    if not Path(csv_path).exists():
        print(f"Cache file {csv_path} not found. Please run benchmarks first.")
        return None, None
    
    cudaonly_data = {}
    tensorcore_data = {}
    
    with open(csv_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            kernel_size = int(row['kernel_size'])
            exec_time = float(row['execution_time_ms']) if row['execution_time_ms'] != 'None' else None
            
            if row['schedule'] == 'cudaonly':
                cudaonly_data[kernel_size] = exec_time
            elif row['schedule'] == 'tensorcore':
                tensorcore_data[kernel_size] = exec_time
    
    # Convert to lists in the same order as kernel_sizes
    cudaonly_times = [cudaonly_data.get(k, None) for k in kernel_sizes]
    tensorcore_times = [tensorcore_data.get(k, None) for k in kernel_sizes]
    
    print(f"Results loaded from {csv_path}")
    return cudaonly_times, tensorcore_times

def collect_benchmark_data(benchmark_name, kernel_sizes, bin_path="./build", use_cache=False, cache_dir="./cache"):
    """
    Collect benchmark data either by running benchmarks or loading from cache.
    """
    if use_cache:
        return load_benchmark_results_from_csv(benchmark_name, kernel_sizes, cache_dir)
    
    # Run benchmarks as before
    schedules = ['cudaonly', 'tensorcore']
    data = {}
    
    for schedule in schedules:
        times = []
        for kernel_size in kernel_sizes:
            print(f"Running {schedule} with kernel size {kernel_size}...")
            time = run_benchmark(benchmark_name, schedule, kernel_size, bin_path)
            times.append(time)
        data[schedule] = times
    
    cudaonly_times, tensorcore_times = data['cudaonly'], data['tensorcore']
    
    # Save results to cache
    save_benchmark_results_to_csv(benchmark_name, kernel_sizes, cudaonly_times, tensorcore_times, cache_dir)
    
    return cudaonly_times, tensorcore_times

def save_multi_benchmark_results_to_csv(benchmarks, kernel_sizes, data, cache_dir="./cache"):
    """
    Save results from multiple benchmarks to a single CSV file.
    """
    # Create cache directory if it doesn't exist
    Path(cache_dir).mkdir(exist_ok=True)
    
    csv_path = f"{cache_dir}/multi_benchmark_results.csv"
    
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(['benchmark_name', 'schedule', 'kernel_size', 'execution_time_ms'])
        
        for benchmark in benchmarks:
            # Write CUDA results
            for kernel_size, time in zip(kernel_sizes, data[benchmark]['cudaonly']):
                writer.writerow([benchmark, 'cudaonly', kernel_size, time])
            
            # Write TensorCore results  
            for kernel_size, time in zip(kernel_sizes, data[benchmark]['tensorcore']):
                writer.writerow([benchmark, 'tensorcore', kernel_size, time])
    
    print(f"Multi-benchmark results saved to {csv_path}")

def load_multi_benchmark_results_from_csv(benchmarks, kernel_sizes, cache_dir="./cache"):
    """
    Load results from multiple benchmarks from a CSV file.
    """
    csv_path = f"{cache_dir}/multi_benchmark_results.csv"
    
    if not Path(csv_path).exists():
        print(f"Cache file {csv_path} not found. Please run benchmarks first.")
        return None
    
    data = {benchmark: {'cudaonly': {}, 'tensorcore': {}} for benchmark in benchmarks}
    
    with open(csv_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            benchmark_name = row['benchmark_name']
            if benchmark_name not in benchmarks:
                continue
                
            kernel_size = int(row['kernel_size'])
            exec_time = float(row['execution_time_ms']) if row['execution_time_ms'] != 'None' else None
            schedule = row['schedule']
            
            data[benchmark_name][schedule][kernel_size] = exec_time
    
    # Convert to the expected format
    for benchmark in benchmarks:
        cudaonly_times = [data[benchmark]['cudaonly'].get(k, None) for k in kernel_sizes]
        tensorcore_times = [data[benchmark]['tensorcore'].get(k, None) for k in kernel_sizes]
        data[benchmark] = {
            'cudaonly': cudaonly_times,
            'tensorcore': tensorcore_times
        }
    
    print(f"Multi-benchmark results loaded from {csv_path}")
    return data

def plot_performance_comparison(title, kernel_sizes, cudaonly_times, tensorcore_times, save_path=None):
    """
    Create a line plot comparing the performance of two schedules.
    """
    plt.figure(figsize=(12, 8))
    
    # Plot lines
    plt.plot(kernel_sizes, tensorcore_times, 's-', linewidth=2, markersize=6, 
             label='Tensor Cores schedules', color='#ff7f0e', alpha=0.8)
    plt.plot(kernel_sizes, cudaonly_times, 'o-', linewidth=2, markersize=6, 
             label='CUDA-only schedules', color='#1f77b4', alpha=0.8)
    
    # Customize the plot
    plt.xlabel('Kernel Size', fontsize=25)
    plt.ylabel('Execution Time (ms)', fontsize=25)
    plt.title(f'{title} Performance Comparison', fontsize=28, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Set x-axis to show all kernel sizes
    plt.xticks(kernel_sizes[::3], fontsize=20)  # Show every other size to avoid crowding
    # Make y-axis start from 0 for better comparison
    ymin, ymax = 0, max(np.max(cudaonly_times), np.max(tensorcore_times)) * 1.05
    xticks = [8,32,56,96,160,256]
    plt.xticks(xticks, xticks, fontsize=20)
    plt.yticks(list(np.arange(ymin, int(np.ceil(ymax)), 0.2)), fontsize=20)
    plt.legend(fontsize=20)
    
    # Add some styling
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def do_benchmark_line_plot(benchmark_name, title, kernel_sizes, use_cache=False, cache_dir="./cache"):
    print("Collecting benchmark data...")
    kernel_sizes = list(kernel_sizes)
    cudaonly_times, tensorcore_times = collect_benchmark_data(
        benchmark_name, kernel_sizes, use_cache=use_cache, cache_dir=cache_dir
    )
    
    if cudaonly_times is None or tensorcore_times is None:
        print("Failed to collect benchmark data.")
        return
    
    # Filter out None values (failed benchmarks)
    valid_data = [(k, c, t) for k, c, t in zip(kernel_sizes, cudaonly_times, tensorcore_times) 
                    if c is not None and t is not None]
    print(valid_data)
    if not valid_data:
        print("No valid benchmark data collected.")
        return
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
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create bars
    bars2 = ax.bar(x - width/2, tensorcore_values, width,
                  label='Tensor Cores schedules', color='#ff7f0e', alpha=0.8)
    bars1 = ax.bar(x + width/2, cudaonly_values, width, 
                  label='CUDA-only schedules', color='#1f77b4', alpha=0.8)
    
    # Customize the plot
    ax.set_xlabel(f'Benchmark', fontsize=25)
    ax.set_ylabel('Execution Time (ms)', fontsize=25)
    ax.set_title(f'Microbenchmark Performance Comparison (k={kernel_sizes[idx]})', fontsize=28, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=25)
    ax.tick_params(axis="y", labelsize=20)
    ax.legend(fontsize=20, loc='upper left')
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

def do_benchmark_bar_chart(use_cache=False, cache_dir="./cache"):
    """
    Create two bar charts comparing performance across conv2d, upsample, and downsample
    - one for kernel size 16 and one for kernel size 32.
    """
    benchmarks = ['conv2d', 'downsample', 'upsample']
    kernel_sizes = [16, 32]
    
    if use_cache:
        print("Loading benchmark data from cache for bar charts...")
        data = load_multi_benchmark_results_from_csv(benchmarks, kernel_sizes, cache_dir)
        if data is None:
            return
    else:
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
        
        # Save multi-benchmark results to cache
        save_multi_benchmark_results_to_csv(benchmarks, kernel_sizes, data, cache_dir)
    
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


def plot_compilation_time(title, kernel_sizes, halide_time, egglog_time, save_path=None):

    plt.figure(figsize=(12, 8))
    
    # Plot lines
    plt.plot(kernel_sizes, egglog_time, 's-', linewidth=2, markersize=6, 
             label='Equality saturation time (egglog)', color='#ff7f0e', alpha=0.8)
    plt.plot(kernel_sizes, halide_time, 'o-', linewidth=2, markersize=6, 
             label='Overall compile time', color='#1f77b4', alpha=0.8)
    
    # Customize the plot
    plt.xlabel('Kernel Size', fontsize=25)
    plt.ylabel('Compile Time (s)', fontsize=25)
    plt.title(f'{title} Compile Time Comparison', fontsize=28, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Set x-axis to show all kernel sizes
    xticks = [8,32,56,96,160,256]
    plt.xticks(xticks, xticks, fontsize=20)
    # Make y-axis start from 0 for better comparison
    ymin, ymax = 0, np.max(halide_time) * 1.05
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

def do_compilation_time_chart():
    def run_and_capture(command, benchmark_name, k):
        """Run a make command, find the .egg file, and copy/rename it."""
        print(f"Running: {command}")
        start_time = time.time()
        proc = subprocess.run(
            command, shell=True, capture_output=True, text=True
        )
        duration = time.time() - start_time
        output = proc.stdout + proc.stderr

        # Look for "Writing egglog program to /tmp/XXX.egg"
        match = re.search(r"Writing egglog program to (/tmp/\S+\.egg)", output)
        if not match:
            print(f"⚠️ No .egg file found for: {command}")
            return

        src_path = match.group(1)
        os.makedirs("benchmark", exist_ok=True)

        dest_name = f"hardboiled_{benchmark_name}_{k}.egg"

        dest_path = os.path.join("benchmark", dest_name)
        shutil.copy(src_path, dest_path)
        print(f"✅ Copied {src_path} -> {dest_path}")
        return duration

    title = "Conv1d"
    egglog_time = []
    halide_time = []
    kernel_sizes = [8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256]
    for k in kernel_sizes:
        # To measure only the last step
        subprocess.run([f"make bin/tensorcore_{k}/conv1d.a 2>/dev/null"], shell=True)
        subprocess.run(["rm", "-f", f"bin/tensorcore_{k}/conv1d.a"])
        halide_duration = run_and_capture(f"make bin/tensorcore_{k}/conv1d.a", "conv1d", k)
        halide_time.append(halide_duration)
        filename = f"benchmark/hardboiled_conv1d_{k}.egg"
        start_time = time.time()
        with open(filename, "r") as f:
            subprocess.run([f"egglog-halide-sidecar < {filename} >/dev/null 2>/dev/null"], shell=True)
        duration = time.time() - start_time
        egglog_time.append(duration)
    
    plot_compilation_time(title, kernel_sizes, halide_time, egglog_time,
                              save_path=f'conv1d_compile_time_comparison.pdf')

def main():
    parser = argparse.ArgumentParser(description='Run benchmarks and generate performance plots')
    parser.add_argument('--use-cache', action='store_true', 
                        help='Use cached benchmark results instead of running new benchmarks')
    parser.add_argument('--cache-dir', default='./cache',
                        help='Directory to store/load cache files (default: ./cache)')
    parser.add_argument('--line-plot', action='store_true',
                        help='Generate line plot for conv1d benchmark')
    parser.add_argument('--bar-chart', action='store_true', 
                        help='Generate bar charts for conv2d, downsample, and upsample')
    parser.add_argument('--all', action='store_true',
                        help='Generate both line plot and bar charts (default if no specific plot is requested)')
    
    args = parser.parse_args()
    # If no specific plot is requested, do all
    if not any([args.line_plot, args.bar_chart]):
        args.all = True
    
    if args.line_plot or args.all:
        print("Generating line plot...")
        kernel_sizes = [8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256]
        do_benchmark_line_plot("conv1d", "Conv1D", kernel_sizes,
                              use_cache=args.use_cache, cache_dir=args.cache_dir)
    
    if args.bar_chart or args.all:
        print("Generating bar charts...")
        do_benchmark_bar_chart(use_cache=args.use_cache, cache_dir=args.cache_dir)

    # if args.all:
    #     do_compilation_time_chart()

if __name__ == "__main__":
    main()
