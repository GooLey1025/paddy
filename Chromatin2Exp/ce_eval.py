#!/usr/bin/env python

import argparse
import yaml
import os

import h5py
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras import mixed_precision

from paddy import bed
from paddy import dataset
from paddy import seqnn
from paddy import trainer
"""
paddy_eval.py

Evaluate the accuracy of a trained model on held-out sequences.
"""


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument(
        "--head",
        dest="head_i",
        default=0,
        type=int,
        help="Parameters head to evaluate [Default: %(default)s]",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        default="eval_out",
        help=
        "Output directory for evaluation statistics [Default: %(default)s]",
    )
    parser.add_argument(
        "--rank",
        default=False,
        action="store_true",
        help="Compute Spearman rank correlation [Default: %(default)s]",
    )
    parser.add_argument(
        "--save",
        default=False,
        action="store_true",
        help="Save targets and predictions numpy arrays [Default: %(default)s]",
    )
    parser.add_argument(
        "--shifts",
        default="0",
        help="Ensemble prediction shifts [Default: %(default)s]",
    )
    parser.add_argument(
        "--step",
        default=1,
        type=int,
        help="Step across positions [Default: %(default)s]",
    )
    parser.add_argument(
        "-m",
        "--mixed-precision",
        default=False,
        action="store_true",
        help="use mixed precision for inference",
    )
    parser.add_argument(
        "-t",
        "--targets_file",
        default=None,
        help="File specifying target indexes and labels in table format",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split label for eg TFR pattern [Default: %(default)s]",
    )
    parser.add_argument(
        "--transpose_input",
        default=False,
        action="store_true",
        help=
        "Transpose input features from [num_bins, num_tracks] to [num_tracks, num_bins] [Default: %(default)s]",
    )

    parser.add_argument("params_file", help="YAML file with model parameters")
    parser.add_argument("model_file", help="Trained model HDF5.")
    parser.add_argument("data_dir", help="Train/valid/test data directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # parse shifts to integers
    args.shifts = [int(shift) for shift in args.shifts.split(",")]

    #######################################################
    # inputs

    # read targets
    if args.targets_file is None:
        args.targets_file = "%s/targets.txt" % args.data_dir
    targets_df = pd.read_csv(args.targets_file, index_col=0, sep="\t")

    # Filter targets to only include the specified split if 'split' column exists
    original_targets_len = len(targets_df)
    if 'split' in targets_df.columns and args.split:
        print(f"Filtering targets to only include {args.split} split")
        targets_df = targets_df[targets_df['split'] == args.split]
        if len(targets_df) == 0:
            raise ValueError(f"No targets found with split={args.split}")
        print(
            f"Found {len(targets_df)} targets in {args.split} split (was {original_targets_len} before filtering)"
        )

    # read model parameters
    with open(args.params_file) as params_open:
        params = yaml.safe_load(params_open)
    params_model = params["model"]
    params_train = params["train"]

    # construct eval data
    eval_data = dataset.TracksDataset(
        args.data_dir,
        split_label=args.split,
        batch_size=params_train["batch_size"],
        mode="eval",
        transpose_input=args.transpose_input,
    )

    ###################
    # mixed precision #
    ###################
    if args.mixed_precision:
        mixed_precision.set_global_policy(
            "mixed_float16")  # first set global policy
        params_model["transpose_input"] = args.transpose_input
        seqnn_model = seqnn.TracksNN(params_model)  # then create model
        seqnn_model.restore(args.model_file, args.head_i)
        seqnn_model.append_activation(
        )  # add additional activation to cast float16 output to float32
    else:
        # initialize model
        params_model["transpose_input"] = args.transpose_input
        seqnn_model = seqnn.TracksNN(params_model)
        seqnn_model.restore(args.model_file, args.head_i)

    # Build ensemble with shifts only (no rc)
    seqnn_model.build_shifts_ensemble(args.shifts)

    #######################################################
    # evaluate
    loss_label = params_train.get("loss", "poisson").lower()
    spec_weight = params_train.get("spec_weight", 1)
    loss_fn = trainer.parse_loss(loss_label, spec_weight=spec_weight)

    # evaluate
    test_loss, test_metric1, test_metric2 = seqnn_model.evaluate(
        eval_data, loss_label=loss_label, loss_fn=loss_fn)

    # Print debugging information about lengths
    print(f"\nDebug - targets_df length: {len(targets_df)}")
    print(f"Debug - test_metric1 length: {len(test_metric1)}")
    print(f"Debug - test_metric2 length: {len(test_metric2)}")

    # print summary statistics
    print("\nTest Loss:         %7.5f" % test_loss)

    # Generate generic tissue names
    num_tissues = len(test_metric1)
    tissue_names = [f"Tissue_{i+1}" for i in range(num_tissues)]

    # Approach 1: Evaluate overall performance per tissue
    if loss_label == "bce":
        print("Test AUROC:        %7.5f" % test_metric1.mean())
        print("Test AUPRC:        %7.5f" % test_metric2.mean())

        # Write tissue-level statistics
        tissue_acc_df = pd.DataFrame({
            "tissue": tissue_names,
            "auroc": test_metric1,
            "auprc": test_metric2
        })

    else:
        print("Test PearsonR:     %7.5f" % test_metric1.mean())
        print("Test R2:           %7.5f" % test_metric2.mean())

        # Write tissue-level statistics
        tissue_acc_df = pd.DataFrame({
            "tissue": tissue_names,
            "pearsonr": test_metric1,
            "r2": test_metric2
        })

    # Save tissue-level metrics
    tissue_acc_df.to_csv("%s/tissue_acc.txt" % args.out_dir,
                         sep="\t",
                         index=False,
                         float_format="%.5f")

    #######################################################
    # Compute predictions for all genes and all tissues

    # compute predictions and get targets
    print("\nComputing predictions for per-gene evaluation...")
    test_preds = seqnn_model.predict(eval_data,
                                     stream=True,
                                     step=args.step,
                                     dtype="float16")
    test_targets = eval_data.numpy(return_inputs=False, step=args.step)

    print(f"Debug - test_preds shape: {test_preds.shape}")
    print(f"Debug - test_targets shape: {test_targets.shape}")

    # Approach 2: Evaluate each gene's performance across all tissues
    # First, extract gene information from targets_df
    gene_ids = targets_df['GeneID'].values

    # Check if we have enough genes
    num_genes = test_preds.shape[0]
    if num_genes > len(gene_ids):
        print(
            f"Warning: More predictions ({num_genes}) than genes in targets_df ({len(gene_ids)})"
        )
        gene_ids = np.append(
            gene_ids, [f"Gene_{i+1}" for i in range(len(gene_ids), num_genes)])
    elif num_genes < len(gene_ids):
        print(
            f"Warning: Fewer predictions ({num_genes}) than genes in targets_df ({len(gene_ids)})"
        )
        gene_ids = gene_ids[:num_genes]

    # glue language here, lol
    # Reshape test_targets to match test_preds shape if needed
    if len(test_targets.shape) == 3 and test_targets.shape[1] == 1:
        print(
            f"Reshaping test_targets from {test_targets.shape} to {(test_targets.shape[0], test_targets.shape[2])}"
        )
        test_targets = np.squeeze(test_targets,
                                  axis=1)  # Remove the middle dimension
        print(f"New test_targets shape: {test_targets.shape}")

    # Compute per-gene correlations
    gene_pearsonr = []
    gene_r2 = []
    gene_spearmanr = [] if args.rank else None

    # Count how many genes have constant values
    constant_genes = 0
    overflow_count = 0

    print("Computing per-gene metrics...")
    for i in tqdm(range(num_genes)):
        # Get predictions and targets for this gene across all tissues
        gene_pred = test_preds[i]
        gene_target = test_targets[i]

        # Check if target or prediction is constant
        if np.std(gene_target) < 1e-10 or np.std(gene_pred) < 1e-10:
            # For constant inputs, set correlation to NaN
            pr = np.nan
            r2 = np.nan
            constant_genes += 1
        else:
            try:
                # Calculate Pearson correlation
                pr, _ = pearsonr(gene_target, gene_pred)

                # Calculate R^2 safely
                ss_tot = np.sum((gene_target - np.mean(gene_target))**2)
                ss_res = np.sum((gene_target - gene_pred)**2)

                # Avoid division by zero or very small numbers
                if ss_tot > 1e-10:
                    r2 = 1 - (ss_res / ss_tot)
                    # Clip R² to a reasonable range to avoid overflow
                    r2 = max(min(r2, 1.0), -1.0)  # Clip to [-1, 1] range
                else:
                    r2 = 0
                    overflow_count += 1
            except Exception as e:
                print(f"Error calculating metrics for gene {i}: {str(e)}")
                pr = np.nan
                r2 = np.nan

        gene_pearsonr.append(pr)
        gene_r2.append(r2)

        # Calculate Spearman correlation if requested
        if args.rank:
            if np.std(gene_target) < 1e-10 or np.std(gene_pred) < 1e-10:
                sr = np.nan
            else:
                try:
                    sr, _ = spearmanr(gene_target, gene_pred)
                except Exception:
                    sr = np.nan
            gene_spearmanr.append(sr)

    # Report statistics on problematic genes
    if constant_genes > 0:
        print(
            f"\nWarning: {constant_genes} genes had constant values (correlation undefined)"
        )
    if overflow_count > 0:
        print(
            f"Warning: {overflow_count} genes had near-zero variance leading to potential overflow in R² calculation"
        )

    # Filter out NaN values for calculating means
    valid_pearsonr = np.array([p for p in gene_pearsonr if not np.isnan(p)])
    valid_r2 = np.array([r for r in gene_r2 if not np.isnan(r)])
    valid_percent = (len(valid_pearsonr) / len(gene_pearsonr)) * 100

    # Create DataFrame for gene-level metrics
    if args.rank:
        gene_acc_df = pd.DataFrame({
            "GeneID": gene_ids,
            "pearsonr": gene_pearsonr,
            "spearmanr": gene_spearmanr,
            "r2": gene_r2
        })
    else:
        gene_acc_df = pd.DataFrame({
            "GeneID": gene_ids,
            "pearsonr": gene_pearsonr,
            "r2": gene_r2
        })

    # Add other gene information if available
    if 'Location' in targets_df.columns and len(gene_acc_df) <= len(
            targets_df):
        gene_acc_df['Location'] = targets_df[
            'Location'].values[:len(gene_acc_df)]

    if 'split' in targets_df.columns and len(gene_acc_df) <= len(targets_df):
        gene_acc_df['split'] = targets_df['split'].values[:len(gene_acc_df)]

    # Save gene-level metrics
    gene_acc_df.to_csv("%s/gene_acc.txt" % args.out_dir,
                       sep="\t",
                       index=False,
                       float_format="%.5f")

    # Print summary of gene-level metrics
    print(f"\nGene-level metrics summary:")
    print(f"Mean Pearson correlation: {np.mean(valid_pearsonr):.5f}")
    print(f"Mean R2: {np.mean(valid_r2):.5f}")
    if args.rank:
        valid_spearmanr = np.array(
            [s for s in gene_spearmanr if not np.isnan(s)])
        print(f"Mean Spearman correlation: {np.mean(valid_spearmanr):.5f}")

    # Print percentage of valid metrics
    print(f"\nValid metrics: {valid_percent:.1f}% of genes")
    print(
        f"Valid gene count: {len(valid_pearsonr)} out of {len(gene_pearsonr)}")

    # Save predictions and targets if requested
    if args.save:
        with h5py.File("%s/preds.h5" % args.out_dir, "w") as preds_h5:
            preds_h5.create_dataset("preds", data=test_preds)
        with h5py.File("%s/targets.h5" % args.out_dir, "w") as targets_h5:
            targets_h5.create_dataset("targets", data=test_targets)


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
