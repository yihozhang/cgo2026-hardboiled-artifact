import subprocess
import os
import sys
import re
from pprint import pprint
from collections import defaultdict

schedules = ["cudaonly", "tensorcore"]

# Define each benchmark as a dictionary with command-line flag names
benchmarks = [
    # Conv1D
    #{"-b": "conv1d", "-conv_k": 8, "-conv_col": 4096, "-conv_row": 4096, "-v": True},
    
    # Conv2D
    #{"-b": "conv2d", "-conv_k": 16, "-conv_col": 4096, "-conv_row": 4096, "-v": True},
    
    # Upsample
    #{"-b": "upsample", "-conv_k": 16, "-conv_col": 4096, "-conv_row": 4096, "-v": True},
    
    # Downsample
    #{"-b": "downsample", "-conv_k": 16, "-conv_col": 4096, "-conv_row": 4096, "-v": True},

    # Denoise
    #{"-b": "denoise", "-conv_col": 4096, "-conv_row": 4096, "-v": True},

    # Resizeg
    #{"-b": "resize", "-conv_col": 4096, "-conv_row": 4096, "-v": True},

    # Rec filter
    #{"-b": "rec_filter", "-conv_col": 2097152, "-v": True},

    # Matmul
    #{"-b": "matmul", "-mm_mnk": [1024, 1024, 1024],  "-v": True},
    #{"-b": "matmul", "-mm_mnk": [4096, 4096, 4096],  "-v": False},

    # Conv Layer (NHWC)
    {"-b": "conv_layer", "-conv_k": 3, "-nhwc": [128, 64, 64, 16], "-v": True},
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
    kernel_size = int(kernel_match.group(1)) if kernel_match else 'N/A'
    
    # Extract image size
    image_match = re.search(r'Image size: (\d+)x(\d+)', output)
    width = int(image_match.group(1)) if image_match else None
    height = int(image_match.group(2)) if image_match else None

    # General input size string captured exactly as printed by the programs
    input_size = None

    # Extract NHWC for conv_layer (use H and W as height/width)
    nhwc_match = re.search(r'NHWC: (\d+)x(\d+)x(\d+)x(\d+)', output)
    if nhwc_match:
        n = int(nhwc_match.group(1))
        h = int(nhwc_match.group(2))
        w = int(nhwc_match.group(3))
        c = int(nhwc_match.group(4))
        width = w
        height = h
        input_size = f"{n}x{h}x{w}x{c}"

    # Extract matrix size
    image_match = re.search(r'Matrix size: (\d+)x(\d+)x(\d+)', output)
    if image_match:
        width = int(image_match.group(1))
        height = int(image_match.group(2))
        input_size = f"{image_match.group(1)}x{image_match.group(2)}x{image_match.group(3)}"
    elif input_size is None:
        # Fall back to Image size if present
        image_dims = re.search(r'Image size: (\d+)x(\d+)', output)
        if image_dims:
            input_size = f"{image_dims.group(1)}x{image_dims.group(2)}"
    
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
    outputs_match = "N/A"
    outputs_match = True if "Outputs match" in output else outputs_match
    outputs_match = False if "Outputs do not match" in output else outputs_match
    
    return {
        'benchmark': benchmark,
        'schedule': schedule,
        'kernel_size': kernel_size,
        'width': width,
        'height': height,
        'input_size': input_size,
        'runtime': runtime,
        'outputs_match': outputs_match
    }

def compute_speedups(results):
    # Group results by configuration (everything except schedule and runtime)
    configs = defaultdict(dict)
    for result in results:
        key = (result['benchmark'], result['kernel_size'], result['input_size'])
        configs[key][result['schedule']] = float(result['runtime'])

    # Compute speedups
    speedups = []
    for (benchmark, ksize, input_size), runtimes in configs.items():
        if 'cudaonly' in runtimes and 'tensorcore' in runtimes:
            cuda_time = runtimes['cudaonly']
            tensor_time = runtimes['tensorcore']
            speedup = cuda_time / tensor_time
            speedups.append({
                'benchmark': benchmark,
                'kernel_size': ksize,
                'input_size': input_size,
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
        elif isinstance(value, list):
            args.append(flag)
            args += [str(v) for v in value]
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
            f.write("hardware,benchmark,schedule,kernel_size,input_size,runtime,outputs_match\n")
        f.write(f"{result['hardware']},{result['benchmark']},{result['schedule']},{result['kernel_size']},{result.get('input_size','')},{result['runtime']},{result['outputs_match']}\n")

def write_speedup_to_csv(speedup, csv_file):
    """Write a single speedup result to the CSV file."""
    # Check if file exists to determine if we need to write header
    file_exists = os.path.exists(csv_file)

    with open(csv_file, "a") as f:
        if not file_exists:
            f.write("hardware,benchmark,kernel_size,input_size,cuda_time,tensor_time,speedup\n")
        f.write(f"{speedup['hardware']},{speedup['benchmark']},{speedup['kernel_size']},{speedup['input_size']},{speedup['cuda_time']},{speedup['tensor_time']},{speedup['speedup']}\n")

def write_log_entry(output, log_file):
    """Write a single log entry to the log file."""
    with open(log_file, "a") as f:
        f.write(output)

def main():
    results = []
    hardware = "A100_80GB"  # Hardware information
    
    # Initialize files
    raw_results_file = "raw_results.csv"
    speedups_file = "speedups.csv"
    log_file = "results.log"
    
    # Clear existing files
    for file in [raw_results_file, speedups_file, log_file]:
        if os.path.exists(file):
            os.remove(file)
    
    # Dictionary to store results for speedup calculation
    config_results = defaultdict(dict)
    
    for benchmark in benchmarks:
        for schedule in schedules:
            # Print the benchmark name, schedule, and full config
            config_str = ', '.join(f"{k}={v}" for k, v in benchmark.items() if k != '-b')
            print(f"=== Running {benchmark['-b']}; Schedule: {schedule}, Config: {config_str} ===\n")
            
            # Step 1: Rebuild binary (note: if compiling for anything other than host, it will build the lib instead of the binary)
            env = os.environ.copy()
            rebuild_args = dict_to_cmd_args(benchmark)
            rebuild_args.extend(["-t", "host", "-s", schedule])
            run_or_exit(["./rebuild.sh"] + rebuild_args, env=env)

            # Step 2: Execute compiled binary
            if benchmark['-v']:
                os.environ['VERIFY_OUTPUT'] = '1'
            else:
                os.environ.pop('VERIFY_OUTPUT', None)
            env = os.environ.copy()
            output = run_or_exit(["./build/" + benchmark['-b']], env=env)
            
            # Write raw output to log file
            write_log_entry(output, log_file)

            # Step 3: Parse the output
            result = parse_benchmark_output(output)
            result['hardware'] = hardware
            results.append(result)
            
            # Write parsed csv to results file
            write_result_to_csv(result, raw_results_file)
            
            # Store result for speedup calculation
            key = (result['benchmark'], result['kernel_size'], result['input_size'])
            config_results[key][result['schedule']] = float(result['runtime'])
            
            # If we have both schedules for this config, calculate and write speedup
            if len(config_results[key]) == 2:
                cuda_time = config_results[key]['cudaonly']
                tensor_time = config_results[key]['tensorcore']
                speedup = cuda_time / tensor_time

                speedup_result = {
                    'hardware': hardware,
                    'benchmark': result['benchmark'],
                    'kernel_size': result['kernel_size'],
                    'input_size': result['input_size'],
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