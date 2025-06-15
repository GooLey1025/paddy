#!/usr/bin/env python3

import yaml
import itertools
import subprocess
import os
import argparse
import concurrent.futures
from datetime import datetime
import copy
import re
import csv
import pandas as pd
import numpy as np


def parse_comma_separated_values(param_grid):
    """Parse comma-separated values in the parameter grid."""
    parsed_grid = {}
    for key, values in param_grid.items():
        if isinstance(values, str) and ',' in values:
            # Split by comma and convert to appropriate type
            parsed_values = [v.strip() for v in values.split(',')]
            # Try to convert to numeric if possible
            parsed_values = [
                float(v) if re.match(r'^-?\d+(\.\d+)?$', v) else v
                for v in parsed_values
            ]
            # Convert to int if it's a whole number
            parsed_values = [
                int(v) if isinstance(v, float) and v.is_integer() else v
                for v in parsed_values
            ]
            parsed_grid[key] = parsed_values
        else:
            parsed_grid[key] = [values]
    return parsed_grid


def run_training(params_file,
                 output_dir,
                 log_dir,
                 experiment_id,
                 tissue_type='23tissues',
                 log_output=True):
    log_file = f"{os.path.dirname(params_file)}/training_output.log"
    redirect = f" > {log_file} 2>&1" if log_output else ""
    cmd = f"./ce_train.py -o {output_dir} -l {log_dir} {params_file} {tissue_type}{redirect}"
    print(
        f"Running experiment {experiment_id}: {os.path.basename(os.path.dirname(params_file))}"
    )
    process = subprocess.run(cmd, shell=True)
    return process.returncode, log_file


def extract_metrics_from_log(log_file):
    """Extract metrics from the log file."""
    metrics = {}

    if not os.path.exists(log_file):
        print(f"Warning: Log file {log_file} does not exist")
        return metrics

    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()

        # Look for the metrics line at the end of the file
        for line in reversed(lines):
            if "best!" in line:
                # Use regex to extract all metrics
                pattern = r'(train_loss|train_r|train_r2|valid_loss|valid_r|valid_r2): ([\d\.]+)'
                matches = re.findall(pattern, line)

                for metric_name, metric_value in matches:
                    metrics[metric_name] = float(metric_value)

                break
    except Exception as e:
        print(f"Error extracting metrics from {log_file}: {str(e)}")

    return metrics


def extract_param_grid_from_yaml(data, prefix=""):
    """Recursively extract comma-separated values from YAML data."""
    param_grid = {}

    if isinstance(data, dict):
        for key, value in data.items():
            current_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                # Recursively scan nested dictionaries
                nested_grid = extract_param_grid_from_yaml(value, current_key)
                param_grid.update(nested_grid)
            elif isinstance(value, list):
                # Handle lists (like trunk parameter)
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        list_key = f"{current_key}.{i}"
                        nested_grid = extract_param_grid_from_yaml(
                            item, list_key)
                        param_grid.update(nested_grid)
            elif isinstance(value, str) and ',' in value:
                # Found comma-separated values
                param_grid[current_key] = value

    return param_grid


def flatten_dict(d, parent_key='', sep='.'):
    """Flatten a nested dictionary into a single-level dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def update_nested_dict(target_dict, key_path, value):
    """Update a nested dictionary using a dot-notation key path."""
    keys = key_path.split('.')

    # Navigate to the right level
    for key in keys[:-1]:
        if key.isdigit():
            # Handle list index
            key = int(key)
        if isinstance(target_dict, list):
            # Ensure list is long enough
            while len(target_dict) <= key:
                target_dict.append({})
            target_dict = target_dict[key]
        else:
            # Handle dictionary key
            if key not in target_dict:
                target_dict[key] = {}
            target_dict = target_dict[key]

    # Set the value at the final level
    final_key = keys[-1]
    if final_key.isdigit():
        final_key = int(final_key)
        if isinstance(target_dict, list):
            while len(target_dict) <= final_key:
                target_dict.append(None)
            target_dict[final_key] = value
        else:
            target_dict[final_key] = value
    else:
        target_dict[final_key] = value


def grid_search(params_file,
                n_parallel=1,
                output_dir=None,
                tissue_type='23tissues',
                resume=False,
                seeds=None):
    """Run grid search for all parameter combinations."""
    # Load the base parameters
    with open(params_file, 'r') as file:
        base_params = yaml.safe_load(file)

    # Extract grid parameters from the YAML file
    param_grid_raw = extract_param_grid_from_yaml(base_params)

    # Parse comma-separated values
    param_grid = parse_comma_separated_values(param_grid_raw)

    if param_grid:
        print(f"Found grid search parameters in YAML: {param_grid}")
    else:
        print(
            "No grid search parameters (comma-separated values) found in YAML file."
        )
        return

    # Generate all combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))

    print(f"Generated {len(combinations)} parameter combinations")

    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir:
        experiment_base_dir = output_dir
    else:
        experiment_base_dir = f"experiments/grid_search_{timestamp}"

    # Check if output directory exists and we're resuming
    resuming = False
    completed_experiments = set()
    experiment_log = []

    if resume and os.path.exists(experiment_base_dir):
        resuming = True
        print(f"Resuming from existing directory: {experiment_base_dir}")

        # Load existing experiment log if available
        log_file = os.path.join(experiment_base_dir, 'experiment_log.yaml')
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                experiment_log = yaml.safe_load(f) or []

            # Track completed experiments
            for entry in experiment_log:
                if entry.get('status') == 'success':
                    completed_experiments.add(entry['experiment'])

            print(f"Found {len(completed_experiments)} completed experiments")

    os.makedirs(experiment_base_dir, exist_ok=True)

    # Prepare experiment configurations
    experiment_configs = []
    all_exp_configs = []  # Store all experiment configs for report generation

    for i, combo in enumerate(combinations):
        # If seeds are specified, create multiple experiments for each combination
        if seeds:
            for seed in seeds:
                exp_name = f"exp_{i}_seed_{seed}"

                # Create a deep copy of base parameters
                current_params = copy.deepcopy(base_params)
                param_desc = []

                # Remove all comma-separated values from the original parameters
                for key in param_grid_raw.keys():
                    keys_path = key.split('.')
                    target = current_params
                    for k in keys_path[:-1]:
                        if k.isdigit():
                            k = int(k)
                        if isinstance(target, list):
                            if k < len(target):
                                target = target[k]
                            else:
                                break
                        elif k in target:
                            target = target[k]
                        else:
                            break

                    # Remove comma-separated values to avoid parsing issues
                    final_key = keys_path[-1]
                    if final_key.isdigit():
                        final_key = int(final_key)

                    if isinstance(target, list):
                        if final_key < len(target) and isinstance(
                                target[final_key],
                                str) and ',' in target[final_key]:
                            target[final_key] = None
                    elif isinstance(target, dict):
                        if final_key in target and isinstance(
                                target[final_key],
                                str) and ',' in target[final_key]:
                            target[final_key] = None

                # Update parameters with current combination
                for param_key, param_value in zip(keys, combo):
                    update_nested_dict(current_params, param_key, param_value)
                    param_desc.append(f"{param_key}={param_value}")

                # Add seed to parameters
                if 'train' not in current_params:
                    current_params['train'] = {}
                current_params['train']['seed'] = seed

                # Create experiment directory and save parameters
                exp_dir = os.path.join(experiment_base_dir, exp_name)
                os.makedirs(exp_dir, exist_ok=True)

                params_file_path = os.path.join(exp_dir, "params.yaml")
                with open(params_file_path, 'w') as f:
                    yaml.dump(current_params, f, default_flow_style=False)

                output_dir_path = os.path.join(exp_dir, "train_out")
                log_dir_path = os.path.join(exp_dir, "tensorboard_logs")

                config = {
                    'id': len(experiment_configs) + 1,
                    'total': len(combinations) * len(seeds),
                    'params_file': params_file_path,
                    'output_dir': output_dir_path,
                    'log_dir': log_dir_path,
                    'param_desc': param_desc,
                    'exp_name': exp_name,
                    'params': current_params,
                    'grid_params': dict(zip(keys, combo)),
                    'seed': seed
                }

                all_exp_configs.append(config)

                # Skip already completed experiments for running
                if not (resuming and exp_name in completed_experiments):
                    experiment_configs.append(config)
        else:
            # Original behavior without seeds
            exp_name = f"exp_{i}"

            # Create a deep copy of base parameters
            current_params = copy.deepcopy(base_params)
            param_desc = []

            # Remove all comma-separated values from the original parameters
            for key in param_grid_raw.keys():
                keys_path = key.split('.')
                target = current_params
                for k in keys_path[:-1]:
                    if k.isdigit():
                        k = int(k)
                    if isinstance(target, list):
                        if k < len(target):
                            target = target[k]
                        else:
                            break
                    elif k in target:
                        target = target[k]
                    else:
                        break

                # Remove comma-separated values to avoid parsing issues
                final_key = keys_path[-1]
                if final_key.isdigit():
                    final_key = int(final_key)

                if isinstance(target, list):
                    if final_key < len(target) and isinstance(
                            target[final_key],
                            str) and ',' in target[final_key]:
                        target[final_key] = None
                elif isinstance(target, dict):
                    if final_key in target and isinstance(
                            target[final_key],
                            str) and ',' in target[final_key]:
                        target[final_key] = None

            # Update parameters with current combination
            for param_key, param_value in zip(keys, combo):
                update_nested_dict(current_params, param_key, param_value)
                param_desc.append(f"{param_key}={param_value}")

            # Create experiment directory and save parameters
            exp_dir = os.path.join(experiment_base_dir, exp_name)
            os.makedirs(exp_dir, exist_ok=True)

            params_file_path = os.path.join(exp_dir, "params.yaml")
            with open(params_file_path, 'w') as f:
                yaml.dump(current_params, f, default_flow_style=False)

            output_dir_path = os.path.join(exp_dir, "train_out")
            log_dir_path = os.path.join(exp_dir, "tensorboard_logs")

            config = {
                'id': i + 1,
                'total': len(combinations),
                'params_file': params_file_path,
                'output_dir': output_dir_path,
                'log_dir': log_dir_path,
                'param_desc': param_desc,
                'exp_name': exp_name,
                'params': current_params,
                'grid_params': dict(zip(keys, combo))
            }

            all_exp_configs.append(config)

            # Skip already completed experiments for running
            if not (resuming and exp_name in completed_experiments):
                experiment_configs.append(config)

    if not experiment_configs:
        print("All experiments are already completed. Nothing to run.")
    else:
        # Run experiments in parallel or sequentially
        if n_parallel > 1:
            run_parallel_experiments(experiment_configs, keys, combinations,
                                     experiment_log, experiment_base_dir,
                                     n_parallel, tissue_type)
        else:
            run_sequential_experiments(experiment_configs, keys, combinations,
                                       experiment_log, experiment_base_dir,
                                       tissue_type)

    # Final log update
    with open(os.path.join(experiment_base_dir, 'experiment_log.yaml'),
              'w') as f:
        yaml.dump(experiment_log, f, default_flow_style=False)

    # Generate CSV report with metrics and parameters
    generate_csv_report(experiment_log, all_exp_configs, experiment_base_dir,
                        seeds)

    print(
        f"\nAll experiments completed. Results logged to {os.path.join(experiment_base_dir, 'experiment_log.yaml')}"
    )
    print(
        f"CSV report generated at {os.path.join(experiment_base_dir, 'exp_results.csv')}"
    )


def generate_csv_report(experiment_log,
                        experiment_configs,
                        base_dir,
                        seeds=None):
    """Generate a CSV report with metrics and parameters."""
    # Prepare data for CSV
    report_data = []

    # Map experiment names to configs for easy lookup
    exp_config_map = {
        config['exp_name']: config
        for config in experiment_configs
    }

    if seeds:
        # Group experiments by their base name (without seed)
        exp_groups = {}
        for entry in experiment_log:
            exp_name = entry['experiment']
            base_exp_name = '_'.join(
                exp_name.split('_')[:-2])  # Remove _seed_X

            if base_exp_name not in exp_groups:
                exp_groups[base_exp_name] = []
            exp_groups[base_exp_name].append(entry)

        # Process each group
        for base_exp_name, group_entries in exp_groups.items():
            # Get the first config for this group (they all have the same parameters)
            config = exp_config_map.get(f"{base_exp_name}_seed_{seeds[0]}")
            if not config:
                continue

            # Calculate mean and std for valid_r
            valid_r_values = []
            for entry in group_entries:
                metrics = entry.get('metrics', {})
                if 'valid_r' in metrics:
                    valid_r_values.append(metrics['valid_r'])

            # Create a row with experiment name and aggregated metrics
            row = {'experiment': base_exp_name}

            # Add valid_r mean and std
            if valid_r_values:
                row['valid_r_mean'] = np.mean(valid_r_values)
                row['valid_r_std'] = np.std(valid_r_values)
                # Add individual seed results
                for i, value in enumerate(valid_r_values):
                    row[f"valid_r_seed_{seeds[i]}"] = value

            # Add grid search parameters
            for param_name, param_value in config['grid_params'].items():
                row[f"grid_{param_name}"] = param_value

            # Add all other parameters
            all_params = flatten_dict(config['params'])
            for param_name, param_value in all_params.items():
                if param_name not in config[
                        'grid_params'] and param_name != 'train.seed':
                    row[param_name] = param_value

            report_data.append(row)
    else:
        # Original behavior without seeds
        for entry in experiment_log:
            exp_name = entry['experiment']
            config = exp_config_map.get(exp_name)

            if not config:
                continue

            # Get the log file path
            log_file = f"{os.path.dirname(config['params_file'])}/training_output.log"

            # Extract metrics from log
            metrics = extract_metrics_from_log(log_file)

            # Flatten parameters
            all_params = flatten_dict(config['params'])

            # Create a row with experiment name, metrics, and parameters
            row = {'experiment': exp_name}

            # Add valid_r metric
            if 'valid_r' in metrics:
                row['valid_r'] = metrics['valid_r']

            # Add grid search parameters first
            for param_name, param_value in config['grid_params'].items():
                row[f"grid_{param_name}"] = param_value

            # Add all other parameters
            for param_name, param_value in all_params.items():
                # Skip parameters that are already added as grid parameters
                if param_name not in config['grid_params']:
                    row[param_name] = param_value

            report_data.append(row)

    # Convert to DataFrame for easy sorting and writing to CSV
    if report_data:
        df = pd.DataFrame(report_data)

        # Sort by valid_r_mean if available, otherwise by valid_r
        if 'valid_r_mean' in df.columns:
            df = df.sort_values(by='valid_r_mean', ascending=False)
        elif 'valid_r' in df.columns:
            df = df.sort_values(by='valid_r', ascending=False)

        # Reorder columns: experiment, valid_r metrics, grid params, other params
        metric_cols = []
        if seeds:
            # For seeded experiments, organize valid_r metrics
            metric_cols = ['valid_r_mean', 'valid_r_std']
            for seed in seeds:
                if f"valid_r_seed_{seed}" in df.columns:
                    metric_cols.append(f"valid_r_seed_{seed}")
        else:
            # For non-seeded experiments, just use valid_r
            metric_cols = ['valid_r']

        grid_param_cols = [
            col for col in df.columns if col.startswith('grid_')
        ]
        other_param_cols = [
            col for col in df.columns
            if not col.startswith(('valid_r', 'grid_')) and col != 'experiment'
        ]

        column_order = ['experiment'
                        ] + metric_cols + grid_param_cols + other_param_cols
        df = df[column_order]

        # Write to CSV
        csv_path = os.path.join(base_dir, 'exp_results.csv')
        df.to_csv(csv_path, index=False)
    else:
        print("Warning: No data available for CSV report")


def run_parallel_experiments(configs,
                             keys,
                             combinations,
                             experiment_log,
                             base_dir,
                             n_workers,
                             tissue_type='23tissues'):
    """Run experiments in parallel using ProcessPoolExecutor."""
    print(
        f"Running {len(configs)} experiments with {n_workers} parallel workers"
    )

    with concurrent.futures.ProcessPoolExecutor(
            max_workers=n_workers) as executor:
        futures = []
        for config in configs:
            futures.append(
                executor.submit(run_training, config['params_file'],
                                config['output_dir'], config['log_dir'],
                                f"{config['id']}/{config['total']}",
                                tissue_type))

        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            return_code, log_file = future.result()
            config = configs[i]

            # Extract metrics from log
            metrics = extract_metrics_from_log(log_file)

            # Get the base experiment index from the experiment name
            exp_name = config['exp_name']
            if '_seed_' in exp_name:
                base_exp_idx = int(exp_name.split('_')[1])
            else:
                base_exp_idx = int(exp_name.split('_')[1])

            experiment_log.append({
                'experiment':
                exp_name,
                'parameters':
                dict(zip(keys, combinations[base_exp_idx])),
                'status':
                'success' if return_code == 0 else 'failed',
                'output_dir':
                config['output_dir'],
                'log_dir':
                config['log_dir'],
                'metrics':
                metrics
            })


def run_sequential_experiments(configs,
                               keys,
                               combinations,
                               experiment_log,
                               base_dir,
                               tissue_type='23tissues'):
    """Run experiments sequentially."""
    for i, config in enumerate(configs):
        print(f"\nStarting experiment {config['id']}/{config['total']}")
        print(f"Parameters: {', '.join(config['param_desc'])}")

        return_code, log_file = run_training(
            config['params_file'], config['output_dir'], config['log_dir'],
            f"{config['id']}/{config['total']}", tissue_type)

        # Extract metrics from log
        metrics = extract_metrics_from_log(log_file)

        # Get the base experiment index from the experiment name
        exp_name = config['exp_name']
        if '_seed_' in exp_name:
            base_exp_idx = int(exp_name.split('_')[1])
        else:
            base_exp_idx = int(exp_name.split('_')[1])

        experiment_log.append({
            'experiment':
            exp_name,
            'parameters':
            dict(zip(keys, combinations[base_exp_idx])),
            'status':
            'success' if return_code == 0 else 'failed',
            'output_dir':
            config['output_dir'],
            'log_dir':
            config['log_dir'],
            'metrics':
            metrics
        })


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Run grid search for Paddy training')
    parser.add_argument(
        'params_file',
        type=str,
        help="""Path to the YAML parameters file with comma-separated values,
        which means that the parameter is a grid search parameter.""")
    parser.add_argument('-p',
                        '--parallel',
                        type=int,
                        default=1,
                        help="""Number of parallel training jobs,
                        depends on your GPU memory and GPU performance.""")
    parser.add_argument('-o',
                        '--output_dir',
                        type=str,
                        default=None,
                        help="""Base directory for experiment outputs,
                        if not specified, the default is 'experiments/grid_search_<timestamp>'"""
                        )
    parser.add_argument('-t',
                        '--tissue_type',
                        type=str,
                        default='23tissues',
                        help="""Tissue type for training,
                        default is '23tissues'""")
    parser.add_argument(
        '-r',
        '--resume',
        action='store_true',
        help="""Resume from previous run if output directory exists,
                        skipping completed experiments.""")
    parser.add_argument(
        '-s',
        '--seeds',
        type=int,
        nargs='+',
        help="""List of seeds to use for each parameter combination.
        If not specified, no seeds will be used.""")
    args = parser.parse_args()

    grid_search(args.params_file, args.parallel, args.output_dir,
                args.tissue_type, args.resume, args.seeds)


if __name__ == "__main__":
    main()
