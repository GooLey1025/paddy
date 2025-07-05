#!/usr/bin/env python3
import argparse
import os
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
from paddy import dna
import re
import pysam


def parse_args():
    parser = argparse.ArgumentParser(description='Predict gene expression from DNA sequence')
    parser.add_argument(
        "-m",
        "--model_file",
        type=str,
        required=True,
        help="Trained model HDF5 file or directory containing multiple seed models e.g. seed100_model_best.h5")
    parser.add_argument(
        "-f",
        "--fasta_file",
        type=str,
        required=True,
        help="Reference genome in FASTA format")
    parser.add_argument(
        "-b",
        "--bed_file",
        type=str,
        required=True,
        help="BED file with sequence coordinates to predict")
    parser.add_argument(
        "--params_file",
        type=str,
        required=True,
        help="YAML file with model parameters")
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="preds",
        help="Output directory for predictions")
    parser.add_argument(
        "--tissue_names",
        type=str,
        default=None,
        help="Optional file with tissue names (one per line)")
    parser.add_argument(
        "-l",
        "--seq_length",
        type=int,
        default=32768,
        help="Sequence length [Default: 32768]")
    parser.add_argument(
        "--head",
        dest="head_i",
        default=0,
        type=int,
        help="Parameters head to evaluate")
    parser.add_argument(
        "--save_fasta",
        action="store_true",
        help="Save fetched DNA sequences to FASTA files")
    parser.add_argument(
        "--fasta_dir",
        type=str,
        default="fasta_out",
        help="Directory to save FASTA files [Default: fasta_out]")
    return parser.parse_args()


def get_model_files(model_path):
    """Get list of model files from path."""
    if os.path.isfile(model_path):
        return False, [("model_best", model_path)]
    elif os.path.isdir(model_path):
        model_files = glob.glob(os.path.join(model_path, "*.h5"))
        if not model_files:
            raise ValueError(f"No .h5 files found in directory: {model_path}")

        # Extract seed numbers from filenames
        seed_files = []
        for file in natsorted(model_files):
            filename = os.path.basename(file) # e.g. seed100_model_best.h5
            if filename.startswith('seed'):
                # Extract number after 'seed'
                try:
                    seed_num = int(filename[4:].split('_')[0]) # e.g. seed100_model_best.h5 -> 100
                except ValueError:
                    print(f"Warning: Could not parse seed number from {filename}, using default")
                    seed_num = len(seed_files)  # Use index as fallback
            else:
                seed_num = len(seed_files)  # Use index as fallback
            seed_files.append((seed_num, file))

        return True, seed_files
    else:
        raise ValueError(f"Model path does not exist: {model_path}")


def read_bed_file(bed_file):
    """Read sequence coordinates from BED file."""
    sequences = []
    with open(bed_file, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
                
            fields = line.strip().split()
            if len(fields) < 3:
                print(f"Warning: Invalid BED line: {line.strip()}")
                continue
                
            chrom = fields[0]
            start = int(fields[1])
            end = int(fields[2])
            
            # Use additional fields if available
            name = fields[3] if len(fields) > 3 else f"{chrom}:{start}-{end}"
            
            sequences.append((name, chrom, start, end))
    
    if not sequences:
        raise ValueError(f"No valid sequences found in {bed_file}")
    
    return sequences


def fetch_dna(fasta_open, chrm, start, end, seq_idx=None, name=None, save_fasta=False, fasta_dir=None):
    """Fetch DNA when start/end may reach beyond chromosomes and optionally save to FASTA file."""
    
    # Initialize sequence
    seq_len = end - start
    seq_dna = ""
    
    # Add N's for left over reach
    if start < 0:
        seq_dna = "N" * (-start)
        start = 0
    
    # Get DNA
    try:
        seq_dna += fasta_open.fetch(chrm, start, end)
    except Exception as e:
        print(f"Warning: Error fetching {chrm}:{start}-{end}: {e}", file=sys.stderr)
        # If fetch fails, exit
        exit(1)
    
    # Add N's for right over reach
    if len(seq_dna) < seq_len:
        seq_dna += "N" * (seq_len - len(seq_dna))
    
    if save_fasta and fasta_dir is not None:
        os.makedirs(fasta_dir, exist_ok=True)
        
        if name is not None:
            filename = f"{name}.fa"
        else:
            filename = f"{seq_idx}.fa"
        
        filepath = os.path.join(fasta_dir, filename)
        
        with open(filepath, 'w') as f:
            header = f">{chrm}:{start}-{end}"
            if name and name != f"{chrm}:{start}-{end}":
                header += f" {name}"
            f.write(f"{header}\n")
            
            # Write sequence with line breaks every 60 characters
            for i in range(0, len(seq_dna), 60):
                f.write(f"{seq_dna[i:i+60]}\n")
        
        print(f"Saved sequence to {filepath}")
    
    return seq_dna


def plot_tissue_expression(seq_id, predictions, tissue_names, output_dir):
    """
    Plot prediction values for a sequence across tissues.
    X-axis: tissues
    Y-axis: predicted expression level
    Different lines = different seeds
    """
    pattern = re.compile(r'tissue_(\d+)_seed_(\d+)')
    tissue_seed_map = {}

    for key in predictions:
        match = pattern.fullmatch(key)
        if match:
            tissue_id, seed_id = match.groups()
            tissue_id = int(tissue_id)
            seed_id = int(seed_id)
            tissue_seed_map.setdefault(seed_id, {})[tissue_id] = predictions[key]

    tissue_ids = natsorted(set(t for seed in tissue_seed_map.values() for t in seed))
    seed_ids = natsorted(tissue_seed_map.keys())

    if tissue_names and len(tissue_names) >= len(tissue_ids):
        x_labels = [tissue_names[i] for i in tissue_ids]
    else:
        x_labels = [f'tissue_{i}' for i in tissue_ids]
    
    x_indices = np.arange(len(x_labels))

    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    sns.set_palette("tab10", len(seed_ids))

    for seed_idx in seed_ids:
        y_vals = [tissue_seed_map[seed_idx].get(t, np.nan) for t in tissue_ids]
        plt.plot(x_indices,
                 y_vals,
                 marker='o',
                 label=f'Seed {seed_idx}',
                 linewidth=2,
                 alpha=0.8)

    for i, t in enumerate(tissue_ids):
        mean_key = f'tissue_{t}_mean'
        std_key = f'tissue_{t}_std'
        if mean_key in predictions and std_key in predictions:
            mean_val = predictions[mean_key]
            std_val = predictions[std_key]
            plt.fill_between([i - 0.1, i + 0.1], [mean_val - std_val] * 2,
                             [mean_val + std_val] * 2,
                             color='gray',
                             alpha=0.15)

    plt.xticks(ticks=x_indices, labels=x_labels, rotation=45, ha='right')
    plt.ylabel("Predicted Expression")
    plt.xlabel("Tissue")
    plt.title(f"Sequence: {seq_id} — Expression Prediction Across Tissues")
    plt.legend(title="Seed", bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"{seq_id}_expression.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.save_fasta:
        os.makedirs(args.fasta_dir, exist_ok=True)
        print(f"FASTA files will be saved to: {args.fasta_dir}")

    with open(args.params_file) as params_open:
        params = yaml.safe_load(params_open)

    tissue_names = None
    if args.tissue_names and os.path.exists(args.tissue_names):
        with open(args.tissue_names, 'r') as f:
            tissue_names = [line.strip() for line in f if line.strip()]

    seed_model_bool, model_files = get_model_files(args.model_file)

    sequences = read_bed_file(args.bed_file)
    print(f"Found {len(sequences)} sequences in BED file")

    seq_length = args.seq_length
    if not seq_length:
        seq_length = params["model"].get("seq_length", 32768)
    
    all_predictions = {}
    for seq_id, _, _, _ in sequences:
        all_predictions[seq_id] = {'sequenceID': seq_id}

    fasta_open = pysam.Fastafile(args.fasta_file)

    for seq_idx, (seq_id, chrom, start, end) in enumerate(sequences):
        print(f"\nProcessing sequence {seq_idx+1}/{len(sequences)}: {seq_id} ({chrom}:{start}-{end})")
        
        coord_length = end - start
        
        # Check if coordinates match expected sequence length
        if coord_length != seq_length:
            print(f"Warning: Sequence length from coordinates ({coord_length}) doesn't match model input size ({seq_length}).", file=sys.stderr)
            print(f"Will fetch sequence from {chrom}:{start}-{end} and adjust as needed.", file=sys.stderr)
        
        # Fetch DNA sequence
        seq_dna = fetch_dna(
            fasta_open, 
            chrom, 
            start, 
            end, 
            seq_idx=seq_idx,
            name=seq_id, 
            save_fasta=args.save_fasta, 
            fasta_dir=args.fasta_dir
        )
        
        # Convert to one-hot encoding using paddy.dna module
        seq_1hot = dna.dna_1hot(seq_dna, seq_len=seq_length)
        
        # Add batch dimension
        seq_1hot = np.expand_dims(seq_1hot, axis=0)
        
        # Run predictions for each model
        for model_idx, (seed_num, model_file) in enumerate(model_files):
            print(f"Loading model {model_idx+1}/{len(model_files)} (seed {seed_num}): {os.path.basename(model_file)}")
            
            # Initialize model
            params["model"]["verbose"] = False
            seqnn_model = seqnn.SeqNN(params["model"])
            seqnn_model.restore(model_file, args.head_i)
            
            # Make prediction
            print("Making prediction...")
            pred = seqnn_model(seq_1hot)[0]
            
            # Add expression values for this model
            if len(pred.shape) > 0:
                # Multi-tissue prediction
                feature_count = pred.shape[0]
                for k in range(feature_count):
                    col_name = f'tissue_{k}_seed_{seed_num}'
                    all_predictions[seq_id][col_name] = float(pred[k])
            else:
                # Single value prediction
                col_name = f'expression_seed_{seed_num}'
                all_predictions[seq_id][col_name] = float(pred)
            
            # Clear model to free memory
            del seqnn_model
            tf.keras.backend.clear_session()
    
    # Close reference genome
    fasta_open.close()
    
    # Calculate mean and std for each tissue across seeds
    print("\nCalculating statistics across seeds...")
    for seq_id in all_predictions:
        pred_dict = all_predictions[seq_id]
        
        # Find all tissue columns for this sequence
        tissue_cols = [col for col in pred_dict.keys() if col.startswith('tissue_')]
        if tissue_cols:
            # Get base tissue names (without seed suffix)
            base_tissues = set(col.split('_seed_')[0] for col in tissue_cols)
            
            # Calculate mean and std for each tissue
            for base_tissue in base_tissues:
                seed_values = [pred_dict[col] for col in tissue_cols if col.startswith(base_tissue)]
                pred_dict[f"{base_tissue}_mean"] = np.mean(seed_values)
                pred_dict[f"{base_tissue}_std"] = np.std(seed_values)
    
    # Generate plots
    print("\nGenerating expression plots...")
    if seed_model_bool:
        for seq_id in tqdm(all_predictions):
            plot_tissue_expression(seq_id, all_predictions[seq_id], tissue_names, args.output_dir)
    
    # Convert to DataFrame and save
    print(f"\nSaving results to {os.path.join(args.output_dir, 'predictions.tsv')}")
    
    df = pd.DataFrame(list(all_predictions.values()))
    
    # Reorder columns: sequenceID, mean/std columns, individual seed columns
    mean_std_cols = [col for col in df.columns if col.endswith(('_mean', '_std'))]
    seed_cols = [col for col in df.columns if '_seed_' in col]
    other_cols = [col for col in df.columns if col not in mean_std_cols + seed_cols]
    
    column_order = other_cols + natsorted(mean_std_cols) + natsorted(seed_cols)
    df = df[column_order]
    
    if not seed_model_bool:
        mean_cols = [col for col in df.columns if col.startswith("tissue_") and col.endswith("_mean")]
        rename_dict = {col: col.replace("_mean", "") for col in mean_cols}
        df = df[["sequenceID"] + mean_cols].rename(columns=rename_dict)
    
    df.to_csv(os.path.join(args.output_dir, 'predictions.tsv'), sep='\t', index=False)
    
    print(f"Saved predictions for {len(df)} sequences.")
    print(f"Results and plots saved in: {args.output_dir}")
    
    if args.save_fasta:
        print(f"Saving FASTA files to dir: {args.fasta_dir}")


if __name__ == "__main__":
    main()
