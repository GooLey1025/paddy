# utils.py

import os
import re
import yaml
import subprocess
import sys
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import time

def set_gpu_device(gpu_id=None):
    """
    Set the GPU device(s) to be used before importing TensorFlow.
    
    This function should be called before importing TensorFlow to ensure
    that TensorFlow only allocates memory on the specified GPU(s).
    
    Args:
        gpu_id (str or list, optional): GPU indices to use, either as a 
            comma-separated string or a list of integers.
            If None, will try to parse from command line arguments.
            Defaults to None.
            
    Returns:
        str: The GPU indices being used.
    
    Example:
        # Option 1: Call directly with GPU ID
        set_gpu_device("0,1")
        import tensorflow as tf
        
        # Option 2: Parse from command line
        set_gpu_device()  # Will look for -g or --gpu in sys.argv
        import tensorflow as tf
    """
    if gpu_id is None:
        # Try to parse GPU ID from command line arguments
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("-g", "--gpu", "--visible_device", nargs='+', default=None)
        
        # Parse only known args to avoid conflicts with other argument parsers
        args, _ = parser.parse_known_args()
        gpu_id = args.gpu
        
    # Convert list of GPU IDs to comma-separated string
    if isinstance(gpu_id, list):
        gpu_id = ','.join(map(str, gpu_id))
            
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        print(f"Using GPU(s): {gpu_id}")

    return gpu_id

def parse_comma_separated_values(param_grid):
    parsed_grid = {}
    for key, values in param_grid.items():
        if isinstance(values, str) and ',' in values:
            parsed_values = [v.strip() for v in values.split(',')]
            parsed_values = [
                float(v) if re.match(r'^-?\d+(\.\d+)?$', v) else v
                for v in parsed_values
            ]
            parsed_values = [
                int(v) if isinstance(v, float) and v.is_integer() else v
                for v in parsed_values
            ]
            parsed_grid[key] = parsed_values
        else:
            parsed_grid[key] = [values]
    return parsed_grid


def extract_param_grid_from_yaml(data, prefix=""):
    param_grid = {}
    if isinstance(data, dict):
        for key, value in data.items():
            current_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                param_grid.update(
                    extract_param_grid_from_yaml(value, current_key))
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        list_key = f"{current_key}.{i}"
                        param_grid.update(
                            extract_param_grid_from_yaml(item, list_key))
            elif isinstance(value, str) and ',' in value:
                param_grid[current_key] = value
    return param_grid


def update_nested_dict(target_dict, key_path, value):
    keys = key_path.split('.')
    for key in keys[:-1]:
        key = int(key) if key.isdigit() else key
        if isinstance(target_dict, list):
            while len(target_dict) <= key:
                target_dict.append({})
            target_dict = target_dict[key]
        else:
            target_dict = target_dict.setdefault(key, {})
    final_key = keys[-1]
    final_key = int(final_key) if final_key.isdigit() else final_key
    if isinstance(target_dict, list):
        while len(target_dict) <= final_key:
            target_dict.append(None)
        target_dict[final_key] = value
    else:
        target_dict[final_key] = value


def flatten_dict(d, parent_key='', sep='.'):  # for CSV output
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def extract_metrics_from_log(log_file):
    metrics = {}
    if not os.path.exists(log_file):
        print(f"Warning: Log file {log_file} does not exist")
        return metrics
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
        for line in reversed(lines):
            if "best!" in line:
                pattern = r'(train_loss|train_r|train_r2|valid_loss|valid_r|valid_r2): ([\d\.]+)'
                matches = re.findall(pattern, line)
                for metric_name, metric_value in matches:
                    metrics[metric_name] = float(metric_value)
                break
    except Exception as e:
        print(f"Error extracting metrics from {log_file}: {str(e)}")
    return metrics


def run_training(params_file,
                 output_dir,
                 log_dir,
                 tissue_type=None,
                 log_output=True,
                 script_path=None,
                 advanced_args=''):
    log_file = f"{os.path.dirname(params_file)}/training_output.log"
    print(f"Log file: {log_file}")
    redirect = f" > {log_file} 2>&1" if log_output else ""
    cmd = f"{script_path} {advanced_args}"
    print(f"cmd: {cmd}")
    cmd += f" {params_file} {tissue_type} {redirect}"
    process = subprocess.run(cmd, shell=True)
    return process.returncode, log_file

def legalize_identifier(identifier):
    # Only keep letters, numbers and underscores in identifier for safe filenames
    return re.sub(r'[^A-Za-z0-9_]', '_', str(identifier))

def plot_ism_effects(file_path, tissue_index=12, title=None, save_path=None):
    """
    Plot ISM-style prediction values for a specific tissue across different seeds.
    
    Parameters:
    - file_path: str, path to the input 'predictions.tsv' file
    - tissue_index: int, the tissue index to plot (e.g., 12 for 'tissue_12')
    - title: str, optional plot title
    - save_path: str, optional path to save the figure

    Returns:
    - None
    """
    # Read the input TSV file
    df = pd.read_csv(file_path, sep='\t')
    df.iloc[:, 1:] = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
    # Build column names for different seeds and the mean
    tissue_prefix = f"tissue_{tissue_index}"
    seeds = [100, 200, 300, 400]
    seed_cols = [f"{tissue_prefix}_seed_{s}" for s in seeds]

    # Use provided mean column if available; otherwise calculate average across seeds
    mean_col = f"{tissue_prefix}_mean"
    if mean_col in df.columns:
        avg_values = df[mean_col]
    else:
        avg_values = df[seed_cols].mean(axis=1)
    print("Average values head:")
    print(avg_values.head())
    print("Any NaN in avg_values?", avg_values.isnull().any())
    # Start plotting
    plt.figure(figsize=(12, 4))

    # Plot each seed line if the column exists
    for seed, col in zip(seeds, seed_cols):
        if col in df.columns:
            plt.plot(df.index, df[col], label=f'{tissue_prefix}_seed_{seed}', linewidth=1)

    # Plot average prediction line
    plt.plot(df.index, avg_values, label='Average', color='cyan', linewidth=2)

    # Aesthetic settings
    plt.title(title or f"ISM Plot for {tissue_prefix}", fontsize=12)
    plt.xlabel("Sequence Index")
    plt.ylabel("Prediction Value")
    plt.legend(loc='upper right', ncol=3)
    plt.tight_layout()

    # Save or show the figure
    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()

def exec_par(cmds, max_proc=None, verbose=False):
    """
    Execute the commands in the list 'cmds' in parallel, but
    only running 'max_proc' at a time.
    Args:
        cmds: list of commands to execute
        max_proc: maximum number of processes to run in parallel
        verbose: print command to stderr
    """
    total = len(cmds)
    finished = 0
    running = 0
    p = []

    if max_proc == None:
        max_proc = len(cmds)

    if max_proc == 1:
        while finished < total:
            if verbose:
                print(cmds[finished], file=sys.stderr)
            op = subprocess.Popen(cmds[finished], shell=True)
            os.waitpid(op.pid, 0)
            finished += 1

    else:
        while finished + running < total:
            # launch jobs up to max
            while running < max_proc and finished + running < total:
                if verbose:
                    print(cmds[finished + running], file=sys.stderr)
                p.append(subprocess.Popen(cmds[finished + running], shell=True))
                # print 'Running %d' % p[running].pid
                running += 1

            # are any jobs finished
            new_p = []
            for i in range(len(p)):
                if p[i].poll() != None:
                    running -= 1
                    finished += 1
                else:
                    new_p.append(p[i])

            # if none finished, sleep
            if len(new_p) == len(p):
                time.sleep(1)
            p = new_p

        # wait for all to finish
        for i in range(len(p)):
            p[i].wait()
