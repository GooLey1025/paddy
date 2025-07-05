# utils.py

import os
import re
import yaml
import subprocess
import sys
import argparse

def set_gpu_device(gpu_id=None):
    """
    Set the GPU device(s) to be used before importing TensorFlow.
    
    This function should be called before importing TensorFlow to ensure
    that TensorFlow only allocates memory on the specified GPU(s).
    
    Args:
        gpu_id (str, optional): Comma-separated list of GPU indices to use.
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
        parser.add_argument("-g", "--gpu", "--visible_device", default="0")
        
        # Parse only known args to avoid conflicts with other argument parsers
        args, _ = parser.parse_known_args()
        gpu_id = args.gpu
    
    # Set CUDA_VISIBLE_DEVICES environment variable
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
