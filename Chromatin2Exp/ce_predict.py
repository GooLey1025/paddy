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

from paddy import seqnn


def parse_args():
    parser = argparse.ArgumentParser(description='Predict gene expression')
    parser.add_argument("--model_file",
                        type=str,
                        required=True,
                        help="Trained model HDF5 file")
    parser.add_argument("--h5_file",
                        type=str,
                        required=True,
                        help="Input H5 file with sequences")
    parser.add_argument("--params_file",
                        type=str,
                        required=True,
                        help="YAML file with model parameters")
    parser.add_argument("--output_file",
                        type=str,
                        default=None,
                        help="Output TSV file")
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


def main():
    args = parse_args()

    if args.output_file is None:
        base_name = os.path.basename(args.h5_file).split('.')[0]
        args.output_file = f"{base_name}_predictions.tsv"

    with open(args.params_file) as params_open:
        params = yaml.safe_load(params_open)

    print(f"Initializing model...")
    seqnn_model = seqnn.TracksNN(params["model"])
    seqnn_model.restore(args.model_file, args.head_i)

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

        # Run predictions
        print(f"Making predictions...")
        all_predictions = []

        for i in tqdm(range(0, num_examples, args.batch_size)):
            batch_end = min(i + args.batch_size, num_examples)
            batch_indices = range(i, batch_end)
            batch_sequences = sequence_data[i:batch_end]

            # Make predictions
            batch_preds = seqnn_model.model.predict(batch_sequences, verbose=0)

            for j, idx in enumerate(batch_indices):
                pred = batch_preds[j]

                # Debug prediction shape
                print(f"Prediction shape: {pred.shape}")

                # Create one row per gene with expression values
                row = {
                    'geneID': chrom[idx]  # Use chrom as gene ID
                }

                # Add expression values directly - no averaging needed
                if len(pred.shape) > 0:
                    # If prediction is multi-dimensional, handle it accordingly
                    if len(pred.shape) == 1:
                        # Already flat array of tissue expressions
                        feature_count = pred.shape[0]
                        for k in range(feature_count):
                            row[f'tissue_{k+1}'] = pred[k]
                    elif len(pred.shape) > 1:
                        raise ValueError(
                            f"Prediction shape {pred.shape} is not supported")
                else:
                    # Single value prediction
                    row['expression_0'] = float(pred)

                all_predictions.append(row)

    # Save results
    print(f"Saving results to {args.output_file}")
    df = pd.DataFrame(all_predictions)
    df.to_csv(args.output_file, sep='\t', index=False)

    print(f"Done. Saved expression values for {len(df)} genes.")


if __name__ == "__main__":
    main()
