import matplotlib.pyplot as plt
import numpy as np
import time
import subprocess
import re

def run_and_get_time_other(benchmark, build_args):
    subprocess.run(f"cmake -S . -B build -DCMAKE_PREFIX_PATH=../../halide-install {build_args}", shell=True)
    subprocess.run(f"cmake --build build --target {benchmark}", shell=True)

    result = subprocess.run(f"build/{benchmark}", capture_output=True, text=True, timeout=1000)
    output = result.stdout
        
    time_match = re.search(r'Runtime:\s*([0-9.]+)\n', output)
    if time_match:
        return float(time_match.group(1)) * 1000
    else:
        print(f"Could not parse timing from output: {output}")
        os.system("exit")
    

def run_and_get_time_halide(benchmark, schedule, param):
    print(f"bash ./rebuild.sh -t host -b {benchmark} -s {schedule} {param}")
    subprocess.run(f"bash ./rebuild.sh -t host -b {benchmark} -s {schedule} {param}", shell=True)
    
    result = subprocess.run(f"./build/{benchmark}", capture_output=True, text=True, timeout=1000)
    output = result.stdout
        
    time_match = re.search(r'Runtime:\s*([0-9.]+)\n', output)
    if time_match:
        return float(time_match.group(1)) * 1000
    else:
        print(f"Could not parse timing from output: {output}")
        os.system("exit")

# Organize the benchmark data
benchmark_data = {
    'MatMul\n(1024³)': {
        'Halide (Tensor Cores)': run_and_get_time_halide("matmul", "tensorcore", "-mm_mnk 1024 1024 1024"),
        'Halide (CUDA-only)': run_and_get_time_halide("matmul", "cudaonly", "-mm_mnk 1024 1024 1024"),
        'cuBLASLt': run_and_get_time_other("matmul_cublas", "-DMATMUL_M=1024 -DMATMUL_N=1024 -DMATMUL_K=1024")
    },
    'Conv Layer\n(16 channels)': {
        'Halide (Tensor Cores)': run_and_get_time_halide("conv_layer", "tensorcore", "-conv_k 3 -nhwc 4096 64 64 16"),
        'PyTorch': run_and_get_time_other("conv_layer_pytorch", "-DNN_TENSOR_N=4096 -DNN_TENSOR_H=64 -DNN_TENSOR_W=64 -DNN_TENSOR_C=16 -DKERNEL_SIZE=3"),
        'cuDNN': run_and_get_time_other("conv_layer_cudnn", "-DNN_TENSOR_N=4096 -DNN_TENSOR_H=64 -DNN_TENSOR_W=64 -DNN_TENSOR_C=16 -DKERNEL_SIZE=3")
    },
    'Conv Layer\n(32 channels)': {
        'Halide (Tensor Cores)': run_and_get_time_halide("conv_layer", "tensorcore", "-conv_k 3 -nhwc 4096 64 64 32"),
        'PyTorch': run_and_get_time_other("conv_layer_pytorch", "-DNN_TENSOR_N=4096 -DNN_TENSOR_H=64 -DNN_TENSOR_W=64 -DNN_TENSOR_C=32 -DKERNEL_SIZE=3"),
        'cuDNN': run_and_get_time_other("conv_layer_cudnn", "-DNN_TENSOR_N=4096 -DNN_TENSOR_H=64 -DNN_TENSOR_W=64 -DNN_TENSOR_C=32 -DKERNEL_SIZE=3")
    },
    'Attention\n(N=64, L=4096)': {
        'Halide (Tensor Cores)': run_and_get_time_halide("attention", "tensorcore", "-att 64 4096 64"),
        'PyTorch': run_and_get_time_other("attention_pytorch", "-DATT_D=64 -DATT_L=4096 -DATT_N=64"),
        'Composed Impl.': run_and_get_time_other("attention_cudnn", "-DATT_D=64 -DATT_L=4096 -DATT_N=64"),
    }
}

# Color mapping for implementations - consistent colors across benchmarks
colors = {
    'Halide (CUDA-only)': '#1f77b4',      # tab:blue
    'Halide (Tensor Cores)': '#ff7f0e',     # tab:orange  
    'cuBLASLt': '#2ca02c',                # tab:green
    'PyTorch': '#d62728',                 # tab:red
    'cuDNN': '#9467bd',                   # tab:purple
    'Composed Impl.': '#8c564b'      # tab:brown
}

# Set up the plot
fig, ax = plt.subplots(figsize=(12, 8))

# Position parameters
benchmarks = list(benchmark_data.keys())
x_positions = np.arange(len(benchmarks))

# Plot bars for each benchmark separately
# Keep track of which implementations we've already added to legend
legend_added = set()

for bench_idx, benchmark in enumerate(benchmarks):
    # Get implementations that exist for this benchmark
    benchmark_impls = list(benchmark_data[benchmark].keys())
    benchmark_times = list(benchmark_data[benchmark].values())
    
    # Calculate bar positions for this benchmark
    num_bars = len(benchmark_impls)
    bar_width = 0.8 / num_bars
    start_offset = -bar_width * (num_bars - 1) / 2
    
    # Plot bars for this benchmark
    for bar_idx, (implementation, time) in enumerate(zip(benchmark_impls, benchmark_times)):
        x_pos = x_positions[bench_idx] + start_offset + bar_idx * bar_width
        
        # Only add to legend if we haven't seen this implementation before
        label = implementation if implementation not in legend_added else ""
        if implementation not in legend_added:
            legend_added.add(implementation)
        
        bars = ax.bar(x_pos, time, bar_width,
                      color=colors.get(implementation, '#95A5A6'),
                      alpha=0.8,
                      edgecolor='white', 
                      linewidth=0.5,
                      label=label)
        
        # Add value labels on top of bars
        ax.text(x_pos, time + time * 0.05, f'{time:.2f}', 
               ha='center', va='bottom', fontsize=20, rotation=0)

# Customize the plot
# ax.set_xlabel('Benchmark Operations', fontsize=25, fontweight='bold')
ax.set_ylabel('Time (ms)', fontsize=25, fontweight='bold')
ax.set_title('Performance Comparison on ML workloads', 
             fontsize=28, fontweight='bold', pad=20)

# Set x-axis
ax.set_xticks(x_positions)
ax.set_xticklabels(benchmarks, fontsize=22)

# Use log scale for y-axis due to wide range of values
ax.set_yscale('log')
ax.set_ylim(bottom=0.01)  # Set a reasonable bottom limit for log scale
ymax = np.max([n for v in benchmark_data.values() for n in v.values()])
ax.set_ylim(top=ymax * 2, bottom=0.01)
ax.tick_params(axis="y", labelsize=20)
# Add grid
ax.grid(True, alpha=0.3, axis='y')

# Customize legend
legend = ax.legend(loc='upper left', fontsize=20)
# legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
legend.set_frame_on(True)
legend.get_frame().set_facecolor('white')
legend.get_frame().set_alpha(0.75)

# Adjust layout to prevent legend cutoff
plt.tight_layout()

# Add some styling
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['left'].set_linewidth(0.5)
# ax.spines['bottom'].set_linewidth(0.5)

# Show the plot
# plt.show()

# Optional: Save the plot
plt.savefig('benchmark_comparison.pdf', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
