import subprocess
import os
import sys
import re
from pprint import pprint
from collections import defaultdict

schedules = ["cuda_only", "tensorcore"]

# Define each benchmark as a dictionary with command-line flag names
benchmarks = [
    # Conv1D
    {"-b": "conv1d", "-conv_k": 128, "-conv_col": 3840, "-conv_row": 2160, "-v": True},
    {"-b": "conv1d", "-conv_k": 128, "-conv_col": 7680, "-conv_row": 4320, "-v": False},
    {"-b": "conv1d", "-conv_k": 128, "-conv_col": 15360, "-conv_row": 8640, "-v": False},
    {"-b": "conv1d", "-conv_k": 128, "-conv_col": 30720, "-conv_row": 17280, "-v": False},
    {"-b": "conv1d", "-conv_k": 128, "-conv_col": 61440, "-conv_row": 34560, "-v": False},

    # Conv2D
    {"-b": "conv2d", "-conv_k": 32, "-conv_col": 3840, "-conv_row": 2160, "-v": True},
    {"-b": "conv2d", "-conv_k": 32, "-conv_col": 7680, "-conv_row": 4320, "-v": False},
    {"-b": "conv2d", "-conv_k": 32, "-conv_col": 15360, "-conv_row": 8640, "-v": False},
    {"-b": "conv2d", "-conv_k": 32, "-conv_col": 30720, "-conv_row": 17280, "-v": False},
    {"-b": "conv2d", "-conv_k": 32, "-conv_col": 61440, "-conv_row": 34560, "-v": False},

    # Upsample
    {"-b": "upsample", "-conv_k": 16, "-conv_col": 3840, "-conv_row": 2160, "-v": True},
    {"-b": "upsample", "-conv_k": 16, "-conv_col": 7680, "-conv_row": 4320, "-v": False},
    {"-b": "upsample", "-conv_k": 16, "-conv_col": 15360, "-conv_row": 8640, "-v": False},
    {"-b": "upsample", "-conv_k": 16, "-conv_col": 30720, "-conv_row": 17280, "-v": False},
    {"-b": "upsample", "-conv_k": 16, "-conv_col": 61440, "-conv_row": 34560, "-v": False},

    # Downsample
    {"-b": "downsample", "-conv_k": 16, "-conv_col": 3840, "-conv_row": 2160, "-v": True},
    {"-b": "downsample", "-conv_k": 16, "-conv_col": 7680, "-conv_row": 4320, "-v": False},
    {"-b": "downsample", "-conv_k": 16, "-conv_col": 15360, "-conv_row": 8640, "-v": False},
    {"-b": "downsample", "-conv_k": 16, "-conv_col": 30720, "-conv_row": 17280, "-v": False},
    {"-b": "downsample", "-conv_k": 16, "-conv_col": 61440, "-conv_row": 34560, "-v": False},

    # Matmul
    {"-b": "matmul", "-mm_mnk": 1024,  "-v": True},
    {"-b": "matmul", "-mm_mnk": 2048,  "-v": False},
    {"-b": "matmul", "-mm_mnk": 4096,  "-v": False},
    {"-b": "matmul", "-mm_mnk": 8192,  "-v": False},
    {"-b": "matmul", "-mm_mnk": 16384, "-v": False},
]

def run_or_exit(cmd, env=None):
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print(result.stderr)
        sys.exit(1)
    return result.stdout

def parse_benchmark_output(output):
    # Extract benchmark name and schedule
    running_match = re.search(r'Running (\w+) with:', output)
    benchmark = running_match.group(1) if running_match else None
    
    # Extract kernel size
    kernel_match = re.search(r'Kernel size: (\d+)', output)
    kernel_size = int(kernel_match.group(1)) if kernel_match else None
    
    # Extract image size
    image_match = re.search(r'Image size: (\d+)x(\d+)', output)
    width = int(image_match.group(1)) if image_match else None
    height = int(image_match.group(2)) if image_match else None
    
    # Extract schedule
    schedule_match = re.search(r'Schedule: (\w+)', output)
    schedule = schedule_match.group(1) if schedule_match else None
    
    # Extract runtime and format it to standard decimal notation
    runtime_match = re.search(r'Runtime: ([\d.e-]+)', output)
    if runtime_match:
        runtime = float(runtime_match.group(1))
        # Format to 9 decimal places to ensure we capture small numbers
        runtime = f"{runtime:.9f}"
    else:
        runtime = None
    
    # Check if outputs match
    outputs_match = "Outputs match!" in output
    
    return {
        'benchmark': benchmark,
        'schedule': schedule,
        'kernel_size': kernel_size,
        'width': width,
        'height': height,
        'runtime': runtime,
        'outputs_match': outputs_match
    }

def compute_speedups(results):
    # Group results by configuration (everything except schedule and runtime)
    configs = defaultdict(dict)
    for result in results:
        key = (result['benchmark'], result['kernel_size'], result['width'], result['height'])
        configs[key][result['schedule']] = float(result['runtime'])
    
    # Compute speedups
    speedups = []
    for (benchmark, ksize, width, height), runtimes in configs.items():
        if 'cuda_only' in runtimes and 'tensorcore' in runtimes:
            cuda_time = runtimes['cuda_only']
            tensor_time = runtimes['tensorcore']
            speedup = cuda_time / tensor_time
            speedups.append({
                'benchmark': benchmark,
                'kernel_size': ksize,
                'resolution': f"{width}x{height}",
                'cuda_time': f"{cuda_time:.9f}",
                'tensor_time': f"{tensor_time:.9f}",
                'speedup': f"{speedup:.2f}x"
            })
    
    return speedups

def dict_to_cmd_args(benchmark_dict):
    """Convert a dictionary of flags and values to a list of command-line arguments."""
    args = []
    for flag, value in benchmark_dict.items():
        if flag == "-v" and value:
            args.append(flag)
        elif flag == "-v" and not value:
            continue
        else:
            args.append(flag)
            args.append(str(value))
    return args

def write_result_to_csv(result, csv_file):
    """Write a single result to the CSV file."""
    # Check if file exists to determine if we need to write header
    file_exists = os.path.exists(csv_file)
    
    with open(csv_file, "a") as f:
        if not file_exists:
            f.write("hardware,benchmark,schedule,kernel_size,width,height,runtime,outputs_match\n")
        f.write(f"{result['hardware']},{result['benchmark']},{result['schedule']},{result['kernel_size']},{result['width']},{result['height']},{result['runtime']},{result['outputs_match']}\n")

def write_speedup_to_csv(speedup, csv_file):
    """Write a single speedup result to the CSV file."""
    # Check if file exists to determine if we need to write header
    file_exists = os.path.exists(csv_file)
    
    with open(csv_file, "a") as f:
        if not file_exists:
            f.write("hardware,benchmark,kernel_size,resolution,cuda_time,tensor_time,speedup\n")
        f.write(f"{speedup['hardware']},{speedup['benchmark']},{speedup['kernel_size']},{speedup['resolution']},{speedup['cuda_time']},{speedup['tensor_time']},{speedup['speedup']}\n")

def write_log_entry(output, log_file):
    """Write a single log entry to the log file."""
    with open(log_file, "a") as f:
        f.write(output)

def main():
    results = []
    hardware = "GTX 4090"  # Hardware information
    
    # Initialize files
    raw_results_file = "raw_results.csv"
    speedups_file = "speedups.csv"
    log_file = "results.csv"
    
    # Clear existing files
    #for file in [raw_results_file, speedups_file, log_file]:
        #if os.path.exists(file):
            #os.remove(file)
    
    # Dictionary to store results for speedup calculation
    config_results = defaultdict(dict)
    
    for benchmark in benchmarks:
        for schedule in schedules:
            # Print the benchmark name and schedule
            print(f"=== Running {benchmark['-b']}; Schedule: {schedule}, Config: {benchmark['-conv_k']}, {benchmark['-conv_row']}, {benchmark['-conv_col']}, {benchmark['-v']} ===\n")
            
            # Step 1: Rebuild
            env = os.environ.copy()
            env["HL_DEBUG_CODEGEN"] = "1"
            rebuild_args = dict_to_cmd_args(benchmark)
            rebuild_args.extend(["-t", "win", "-s", schedule])
            run_or_exit(["./rebuild.sh"] + rebuild_args, env=env)

            # Step 2: Execute on remote machine via SSH and capture output
            ssh_args = dict_to_cmd_args(benchmark)
            ssh_args.extend(["-s", schedule])
            output = run_or_exit(["./ssh_win.sh"] + ssh_args)
            
            # Write to log file immediately
            write_log_entry(output, log_file)

            # Step 3: Parse the output
            result = parse_benchmark_output(output)
            result['hardware'] = hardware  # Add hardware information
            results.append(result)
            
            # Write raw result to CSV immediately
            write_result_to_csv(result, raw_results_file)
            
            # Store result for speedup calculation
            key = (result['benchmark'], result['kernel_size'], result['width'], result['height'])
            config_results[key][result['schedule']] = float(result['runtime'])
            
            # If we have both schedules for this config, calculate and write speedup
            if len(config_results[key]) == 2:
                cuda_time = config_results[key]['cuda_only']
                tensor_time = config_results[key]['tensorcore']
                speedup = cuda_time / tensor_time
                speedup_result = {
                    'hardware': hardware,
                    'benchmark': result['benchmark'],
                    'kernel_size': result['kernel_size'],
                    'resolution': f"{result['width']}x{result['height']}",
                    'cuda_time': f"{cuda_time:.9f}",
                    'tensor_time': f"{tensor_time:.9f}",
                    'speedup': f"{speedup:.2f}x"
                }
                write_speedup_to_csv(speedup_result, speedups_file)

            print()
    
    print("\nRaw Results:")
    pprint(results)
    
    print("\nSpeedup Analysis:")
    speedups = compute_speedups(results)
    pprint(speedups)

if __name__ == "__main__":
    main()