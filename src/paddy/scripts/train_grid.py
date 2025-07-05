#! /usr/bin/env python3
# train_grid.py

import argparse
from paddy import grid

def main():
    parser = argparse.ArgumentParser(description="Grid search runner with multiple seeds and resuming support")
    parser.add_argument("-s","--script_path", type=str, required=True, help="Script path to run training")
    parser.add_argument("-a","--advanced_args", type=str, default='', help="Advanced arguments to pass to training script")
    parser.add_argument("params_file", type=str, help="YAML file with grid search parameters")
    parser.add_argument("--output_dir", type=str, default=None, help="Base directory to store experiment outputs")
    parser.add_argument("--seeds", type=int, nargs='+', default=[42], help="List of seeds to use")
    parser.add_argument("--restart", action="store_true", help="Restart from previous run, default is False")
    parser.add_argument("-t","--tissue_type", type=str, default=None, help="Tissue type parameter passed to training")
    parser.add_argument("-p","--parallel_jobs", type=int, default=1, help="Number of parallel jobs to run")
    parser.add_argument("--gpus", type=str, default=None, 
                      help="Comma-separated list of GPU IDs to use (e.g., '0,1,2'). If not specified, all available GPUs will be used.")
    args = parser.parse_args()

    # Set CUDA_VISIBLE_DEVICES if specified
    if args.gpus:
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
        print(f" Using GPUs: {args.gpus}")

    runner = grid.GridSearchRunner(
        base_params_file=args.params_file,
        output_dir=args.output_dir,
        seeds=args.seeds,
        tissue_type=args.tissue_type,
        restart=args.restart,
        script_path=args.script_path,
        advanced_args=args.advanced_args,
        parallel_jobs=args.parallel_jobs
    )

    runner.prepare_experiments()

    total = len(runner.all_experiments)
    pending = len(runner.experiments_to_run)
    done = len(runner.completed_keys)

    print(f" Detected {len(runner.keys)} hyperparameters: {runner.keys}")
    print(f" Total experiment combinations (with seeds): {total}")
    print(f" Previously completed: {done}")
    print(f" Remaining to run: {pending}\n")
    
    if args.parallel_jobs > 1:
        print(f" Running with {args.parallel_jobs} parallel processes")
    
    runner.run()

    print(f"\n All {total} experiments complete. {pending} this time running.")
    print(f" Results saved to: {runner.output_dir}/exp_results.csv")
    print(f" Log saved to: {runner.output_dir}/experiment_log.yaml")

if __name__ == "__main__":
    main()