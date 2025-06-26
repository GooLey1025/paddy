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
    args = parser.parse_args()

    runner = grid.GridSearchRunner(
        base_params_file=args.params_file,
        output_dir=args.output_dir,
        seeds=args.seeds,
        tissue_type=args.tissue_type,
        restart=args.restart,
        script_path=args.script_path,
        advanced_args=args.advanced_args
    )

    runner.prepare_experiments()

    total = len(runner.all_experiments)
    pending = len(runner.experiments_to_run)
    done = len(runner.completed_keys)

    print(f" Detected {len(runner.keys)} hyperparameters: {runner.keys}")
    print(f" Total experiment combinations (with seeds): {total}")
    print(f" Previously completed: {done}")
    print(f" Remaining to run: {pending}\n")
    runner.run()

    print(f"\n All {total} experiments complete. {pending} this time running.")
    print(f" Results saved to: {runner.output_dir}/exp_results.csv")
    print(f" Log saved to: {runner.output_dir}/experiment_log.yaml")

if __name__ == "__main__":
    main()