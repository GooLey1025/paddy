#!/usr/bin/env python3
import argparse
import os
import h5py
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import pandas as pd
import yaml
import sys
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from natsort import natsorted
from paddy import seqnn
import re

def parse_args():
    parser = argparse.ArgumentParser(description='Predict gene expression')
    parser.add_argument(
        "-m",
        "--model_file",
        type=str,
        required=True,
        help=
        "Trained model HDF5 file or directory containing multiple seed models")
    parser.add_argument("-i",
                        "--h5_file",
                        type=str,
                        required=True,
                        help="Input H5 file with sequences")
    parser.add_argument("--params_file",
                        type=str,
                        required=True,
                        help="YAML file with model parameters")
    parser.add_argument("--output_dir",
                        type=str,
                        default="preds",
                        help="Output directory for predictions and plots")
    parser.add_argument("--batch_size",
                        type=int,
                        default=32,
                        help="Batch size for prediction")
    parser.add_argument("--head",
                        dest="head_i",
                        default=0,
                        type=int,
                        help="Parameters head to evaluate")
    return parser.parse_args()


def get_model_files(model_path):
    """Get list of model files from path."""
    if os.path.isfile(model_path):
        return False, [model_path]
    elif os.path.isdir(model_path):
        # Look for h5 files in directory
        model_files = glob.glob(os.path.join(model_path, "*.h5"))
        if not model_files:
            raise ValueError(f"No .h5 files found in directory: {model_path}")
        return True, sorted(model_files)
    else:
        raise ValueError(f"Model path does not exist: {model_path}")



def plot_gene_predictions(gene_id, predictions, output_dir, num_targets):
    """
    Plot prediction values for a gene.
    X-axis: targets (target_1, target_2, ...)
    Y-axis: predicted value
    Different lines = different seeds
    """
    # 自动识别所有 target IDs 和 seed IDs
    pattern = re.compile(r'target_(\d+)_seed_(\d+)')
    target_seed_map = {}

    for key in predictions:
        match = pattern.fullmatch(key)
        if match:
            target_id, seed_id = match.groups()
            target_id = int(target_id)
            seed_id = int(seed_id)
            target_seed_map.setdefault(seed_id, {})[target_id] = predictions[key]

    target_ids = sorted(set(t for seed in target_seed_map.values() for t in seed))
    seed_ids = sorted(target_seed_map.keys())

    x_labels = [f'target_{i}' for i in target_ids]
    x_indices = np.arange(len(x_labels))

    plt.figure(figsize=(14, 6))
    sns.set_style("whitegrid")
    sns.set_palette("tab10", len(seed_ids))

    for seed_idx in seed_ids:
        y_vals = [target_seed_map[seed_idx].get(t, np.nan) for t in target_ids]
        plt.plot(
            x_indices,
            y_vals,
            marker='o',
            label=f'Seed {seed_idx}',
            linewidth=2,
            alpha=0.8
        )

    # 可选：添加 mean ± std 区间
    for i, t in enumerate(target_ids):
        mean_key = f'target_{t}_mean'
        std_key = f'target_{t}_std'
        if mean_key in predictions and std_key in predictions:
            mean_val = predictions[mean_key]
            std_val = predictions[std_key]
            plt.fill_between(
                [i - 0.1, i + 0.1],
                [mean_val - std_val] * 2,
                [mean_val + std_val] * 2,
                color='gray',
                alpha=0.15
            )

    plt.xticks(ticks=x_indices, labels=x_labels, rotation=45, ha='right')
    plt.ylabel("Predicted Expression")
    plt.xlabel("Target")
    plt.title(f"Gene: {gene_id} — Prediction Across Targets by Seed")
    plt.legend(title="Seed", bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"{gene_id}_by_seed.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.params_file) as params_open:
        params = yaml.safe_load(params_open)

    # Get list of model files
    seed_model_bool, model_files = get_model_files(args.model_file)
    print(f"Found {len(model_files)} model files")

    print(f"Reading data from {args.h5_file}")
    with h5py.File(args.h5_file, 'r') as in_h5:
        num_examples = len(in_h5['chrom'])
        print(f"Found {num_examples} examples")

        chrom = in_h5['chrom'][:]  # Using chrom as gene ID
        sequence_data = in_h5['preds'][:]  # Using 'preds' as sequence data
        print(f"Sequence data shape: {sequence_data.shape}")

        # Convert byte strings to regular strings if needed
        if isinstance(chrom[0], bytes):
            chrom = [c.decode('utf-8') for c in chrom]

        # Initialize predictions dictionary
        all_predictions = {}
        for i in range(num_examples):
            all_predictions[chrom[i]] = {'geneID': chrom[i]}

        # Run predictions for each model
        for model_idx, model_file in enumerate(model_files):
            print(
                f"\nLoading model {model_idx + 1}/{len(model_files)}: {os.path.basename(model_file)}"
            )
            seqnn_model = seqnn.TracksNN(params["model"], verbose=False)
            seqnn_model.restore(model_file, args.head_i)

            print(f"Making predictions...")
            for i in tqdm(range(0, num_examples, args.batch_size)):
                batch_end = min(i + args.batch_size, num_examples)
                batch_indices = range(i, batch_end)
                batch_sequences = sequence_data[i:batch_end]

                # Make predictions
                batch_preds = seqnn_model.model.predict(batch_sequences,
                                                        verbose=0)

                for j, idx in enumerate(batch_indices):
                    pred = batch_preds[j]
                    gene_id = chrom[idx]

                    # Add expression values for this model
                    if len(pred.shape) > 0:
                        if len(pred.shape) == 1:
                            # Multi-tissue prediction
                            feature_count = pred.shape[0]
                            for k in range(feature_count):
                                col_name = f'target_{k+1}_seed_{model_idx+1}'
                                all_predictions[gene_id][col_name] = pred[k]
                        elif len(pred.shape) > 1:
                            raise ValueError(
                                f"Prediction shape {pred.shape} is not supported"
                            )
                    else:
                        # Single value prediction
                        col_name = f'expression_0_seed_{model_idx+1}'
                        all_predictions[gene_id][col_name] = float(pred)

            # Clear model to free memory
            del seqnn_model
            tf.keras.backend.clear_session()

        # Calculate mean and std for each target across seeds
        print("\nCalculating statistics across seeds...")
        for gene_id in all_predictions:
            pred_dict = all_predictions[gene_id]

            # Find all target columns for this gene
            target_cols = [
                col for col in pred_dict.keys() if col.startswith('target_')
            ]
            if target_cols:
                # Get base target names (without seed suffix)
                base_targets = set(
                    col.split('_seed_')[0] for col in target_cols)

                # Calculate mean and std for each target
                for base_target in base_targets:
                    seed_values = [
                        pred_dict[col] for col in target_cols
                        if col.startswith(base_target)
                    ]
                    pred_dict[f"{base_target}_mean"] = np.mean(seed_values)
                    pred_dict[f"{base_target}_std"] = np.std(seed_values)

        # Generate plots for each gene
        print("\nGenerating prediction plots...")
        if seed_model_bool:
            # get number of targets from first gene
            num_targets = len(all_predictions[list(all_predictions.keys())[0]]) - 1 # -1 for geneID
            for gene_id in tqdm(all_predictions):
                plot_gene_predictions(gene_id, all_predictions[gene_id],
                                  args.output_dir, num_targets)

    # Convert to DataFrame and save
    print(
        f"\nSaving results to {os.path.join(args.output_dir, 'predictions.tsv')}"
    )
    
    df = pd.DataFrame(list(all_predictions.values()))

    # Reorder columns: geneID, mean/std columns, individual seed columns
    mean_std_cols = [
        col for col in df.columns if col.endswith(('_mean', '_std'))
    ]
    seed_cols = [col for col in df.columns if '_seed_' in col]
    other_cols = [
        col for col in df.columns if col not in mean_std_cols + seed_cols
    ]

    column_order = other_cols + natsorted(mean_std_cols) + natsorted(seed_cols)
    df = df[column_order]

    if not seed_model_bool:
        mean_cols = [col for col in df.columns if col.startswith("target_") and col.endswith("_mean")]
        rename_dict = {col: col.replace("_mean", "") for col in mean_cols}
        df = df[["geneID"] + mean_cols].rename(columns=rename_dict)

    df.to_csv(os.path.join(args.output_dir, 'predictions.tsv'),
              sep='\t',
              index=False)

    print(f"Done. Saved predictions for {len(df)} genes.")
    print(f"Results and plots saved in: {args.output_dir}")


if __name__ == "__main__":
    main()
