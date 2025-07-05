# grid.py

import os
import json
import yaml
import hashlib
import copy
import itertools
import numpy as np
import pandas as pd
from datetime import datetime
from paddy.utils import parse_comma_separated_values, extract_param_grid_from_yaml, update_nested_dict, flatten_dict, extract_metrics_from_log, run_training
from collections import defaultdict
import concurrent.futures

class GridSearchRunner:
    def __init__(self, base_params_file, output_dir=None, seeds=None, tissue_type=None, restart=False, script_path=None, advanced_args='', parallel_jobs=1):
        self.base_params_file = base_params_file
        self.seeds = seeds
        self.tissue_type = tissue_type
        self.restart = restart
        self.script_path = script_path
        self.advanced_args = advanced_args
        self.parallel_jobs = parallel_jobs
        self.output_dir = output_dir or f"experiments/grid_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.output_dir, exist_ok=True)

        self.base_params = self._load_yaml(self.base_params_file)
        self.param_grid_raw = extract_param_grid_from_yaml(self.base_params)
        self.param_grid = parse_comma_separated_values(self.param_grid_raw)
        self.keys = list(self.param_grid.keys())
        self.values = list(self.param_grid.values())
        self.combinations = list(itertools.product(*self.values))
        self.experiment_log_path = os.path.join(self.output_dir, 'experiment_log.yaml')
        self.completed_keys = self._load_completed_keys()
        self.experiments_to_run = []
        self.all_experiments = []

    def _load_yaml(self, file_path):
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _load_completed_log_entries(self):
        if self.restart or not os.path.exists(self.experiment_log_path):
            return []
        with open(self.experiment_log_path, 'r') as f:
            log = yaml.safe_load(f) or []
        return [entry for entry in log if isinstance(entry, dict)]
        

    def _load_completed_keys(self):
        if self.restart or not os.path.exists(self.experiment_log_path):
            return set()
        return set(
            entry['exp_hash_key']
            for entry in self._load_completed_log_entries()
            if entry.get('status') == 'success' and 'exp_hash_key' in entry
        )
    
    def _make_exp_hash_key(self, full_params):
        key_data = {'grid': full_params}
        return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()[:8]

    def prepare_experiments(self):
        for i, combo in enumerate(self.combinations):
            grid_params = dict(zip(self.keys, combo))
            for seed in self.seeds or [None]:
                current_params = copy.deepcopy(self.base_params)
                
                # remove old param_grid's values
                for key in self.param_grid_raw.keys():
                    update_nested_dict(current_params, key, None)
                # add new combination of param values
                for param_key, param_value in grid_params.items():
                    update_nested_dict(current_params, param_key, param_value)

                if seed is not None:
                    current_params['train']['seed'] = seed

                exp_hash_key = self._make_exp_hash_key(full_params=current_params)
  
                exp_name = f"exp_{i}_seed_{seed}" if seed is not None else f"exp_{i}"
                exp_dir = os.path.join(self.output_dir, exp_name)
                os.makedirs(exp_dir, exist_ok=True)
                
                params_path = os.path.join(exp_dir, "params.yaml")
                with open(params_path, 'w') as f:
                    yaml.dump(current_params, f)

                output_dir_path = os.path.join(exp_dir, "train_out")
                log_dir_path = os.path.join(exp_dir, "tensorboard_logs")

                config = {
                    'exp_name': exp_name,
                    'exp_hash_key': exp_hash_key,
                    'params_file': params_path,
                    'output_dir': output_dir_path,
                    'log_dir': log_dir_path,
                    'params': current_params,
                    'grid_params': grid_params,
                    'seed': seed
                }
                self.all_experiments.append(config)
                if exp_hash_key in self.completed_keys:
                    print(f"Skipping {exp_hash_key} because it already exists")
                    continue
                self.experiments_to_run.append(config)


    def run(self):
        if self.parallel_jobs <= 1:
            # Original sequential implementation
            return self._run_sequential()
        else:
            # Parallel implementation
            return self._run_parallel()
    
    def _run_experiment(self, config, job_id=None):
        # Allocate specific GPU if running in parallel
        original_env = None
        if job_id is not None and self.parallel_jobs > 1:
            # Save original environment variable
            original_env = os.environ.get('CUDA_VISIBLE_DEVICES')
            
            # If CUDA_VISIBLE_DEVICES is already set (via --gpus), use those GPUs
            if original_env:
                available_gpus = original_env.split(',')
                if available_gpus:
                    # Round-robin assignment among specified GPUs
                    gpu_id = available_gpus[job_id % len(available_gpus)]
                    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
                    print(f"Assigning experiment {config['exp_name']} to GPU:{gpu_id}")
            else:
                # Auto-detect available GPUs if not specified
                import subprocess
                try:
                    nvidia_smi = subprocess.check_output('nvidia-smi -L', shell=True).decode('utf-8')
                    available_gpus = len([line for line in nvidia_smi.split('\n') if 'GPU ' in line])
                except:
                    available_gpus = 1  # Default to 1 if nvidia-smi fails
                    
                # Assign GPU based on job_id (round-robin)
                if available_gpus > 0:
                    gpu_id = job_id % available_gpus
                    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
                    print(f"Assigning experiment {config['exp_name']} to GPU:{gpu_id}")
        
        print(f"Running: {config['exp_name']}. Total experiments: {len(self.all_experiments)}")
        return_code, log_file = run_training(
            config['params_file'],
            config['output_dir'],
            config['log_dir'],
            tissue_type=self.tissue_type,
            script_path=self.script_path,
            advanced_args=self.advanced_args
        )
        metrics = extract_metrics_from_log(log_file)
        
        # Restore original environment variable if it was changed
        if original_env is not None:
            if original_env:
                os.environ['CUDA_VISIBLE_DEVICES'] = original_env
            else:
                os.environ.pop('CUDA_VISIBLE_DEVICES', None)

        return {
            'experiment': config['exp_name'],
            'params_file': config['params_file'],
            'exp_hash_key': config['exp_hash_key'],
            'parameters': config['grid_params'],
            'status': 'success' if return_code == 0 else 'failed',
            'output_dir': config['output_dir'],
            'log_dir': config['log_dir'],
            'metrics': metrics
        }
    
    def _run_sequential(self):
        experiment_log = []
        for i, config in enumerate(self.experiments_to_run):
            print(f"Running: {config['exp_name']}. Now:{i+1}/{len(self.experiments_to_run)}. All:{len(self.all_experiments)}")
            return_code, log_file = run_training(
                config['params_file'],
                config['output_dir'],
                config['log_dir'],
                tissue_type=self.tissue_type,
                script_path=self.script_path,
                advanced_args=self.advanced_args
            )
            metrics = extract_metrics_from_log(log_file)

            entry = {
                'experiment': config['exp_name'],
                'params_file': config['params_file'],
                'exp_hash_key': config['exp_hash_key'],
                'parameters': config['grid_params'],
                'status': 'success' if return_code == 0 else 'failed',
                'output_dir': config['output_dir'],
                'log_dir': config['log_dir'],
                'metrics': metrics
            }

            old_log = self._load_completed_log_entries()
            full_log = old_log + [entry]
            with open(self.experiment_log_path, 'w') as f:
                yaml.dump(full_log, f)

            self._generate_csv_report(full_log)
    
    def _run_parallel(self):
        print(f"Running {len(self.experiments_to_run)} experiments with {self.parallel_jobs} parallel jobs")
        old_log = self._load_completed_log_entries()
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.parallel_jobs) as executor:
            # Submit jobs with job_id for GPU allocation
            future_to_exp = {}
            for job_id, config in enumerate(self.experiments_to_run):
                future = executor.submit(self._run_experiment, config, job_id)
                future_to_exp[future] = config
            
            for future in concurrent.futures.as_completed(future_to_exp):
                entry = future.result()
                config = future_to_exp[future]
                print(f"Completed: {config['exp_name']}")
                
                # Update log file atomically
                old_log.append(entry)
                with open(self.experiment_log_path, 'w') as f:
                    yaml.dump(old_log, f)
                
                # Generate updated report
                self._generate_csv_report(old_log)

    def _generate_csv_report(self, full_log):
        merged = defaultdict(lambda: {
        'metrics': {}, 
        'valid_r_values': [],
        'parameters': {},
        'experiment': ''
    })

        for entry in full_log:
            if not isinstance(entry, dict):
                continue

            experiment = entry['experiment']
            if '_seed_' in experiment:
                base_name, seed = experiment.rsplit('_seed_', 1)
                suffix = f"seed_{seed}"
            else:
                base_name = experiment
                suffix = "no_seed"

            group = merged[base_name]
            group['experiment'] = base_name

            if not group['parameters']:
                group['parameters'] = entry.get('parameters', {})

            metrics = entry.get('metrics', {})

            if 'valid_r' in metrics:
                group['metrics'][f'valid_r_{suffix}'] = metrics['valid_r']
                group['valid_r_values'].append(metrics['valid_r'])
            if 'train_loss' in metrics:
                group['metrics'][f'train_loss_{suffix}'] = metrics['train_loss']

        report = []
        for group in merged.values():
            row = {}
            row['experiment'] = group['experiment']

            if group['valid_r_values']:
                row['valid_r_mean'] = float(np.mean(group['valid_r_values']))
                row['valid_r_std'] = float(np.std(group['valid_r_values']))

            row.update(group['metrics'])      
            row.update(group['parameters'])   
            report.append(row)

        df = pd.DataFrame(report)

        if 'valid_r_mean' in df.columns:
            df = df.sort_values(by='valid_r_mean', ascending=False)

        front_cols = ['experiment', 'valid_r_mean', 'valid_r_std']
        other_cols = [col for col in df.columns if col not in front_cols]
        df = df[front_cols + other_cols]

        df.to_csv(os.path.join(self.output_dir, 'exp_results.csv'), index=False)

    # def _generate_csv_report(self, full_log):
    #     report = []
    #     for entry in full_log:
    #         if not isinstance(entry, dict):
    #             continue
    #         row = {
    #             'experiment': entry['experiment'],
    #             'status': entry['status']
    #         }
    #         row.update(entry.get('parameters', {}))
    #         row.update(entry.get('metrics', {}))
    #         report.append(row)

    #     df = pd.DataFrame(report)
    #     if 'valid_r' in df.columns:
    #         df = df.sort_values(by='valid_r', ascending=False)
    #     df.to_csv(os.path.join(self.output_dir, 'exp_results.csv'), index=False)
