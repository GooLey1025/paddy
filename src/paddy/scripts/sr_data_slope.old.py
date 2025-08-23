#!/usr/bin/env python3
"""
Generate TFRecord files for SloRice model from SNP data.

This script processes SNP data and generates TFRecord files containing:
- ref_sequence: Reference DNA sequence 
- alt_sequence: Alternate DNA sequence with SNP
- slope: Expression slope value
"""

import argparse
import multiprocessing as mp
import os
import sys
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import pyfaidx
import tensorflow as tf
import yaml
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from paddy.utils import legalize_identifier


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate TFRecord files for SloRice model from SNP data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('snp_file', type=str, help='Excel file containing SNP data')
    parser.add_argument('fasta_file', type=str, help='FASTA file containing reference genome')
    
    # Optional arguments
    parser.add_argument("-s", "--seqs_per_tfr", default=512, type=int, help="Sequences per TFRecord file")
    parser.add_argument("-o", "--output_dir", default="sr_data_out", help="Output directory")
    parser.add_argument("-p", "--processes", default=None, type=int, help="Number of parallel processes")
    parser.add_argument("-l", "--seq_length", default=32768, type=int, help="Sequence length")
    parser.add_argument("--save_fasta", action="store_true", help="Save ref and alt sequences to one fasta file")
    # Data splitting arguments
    parser.add_argument("--valid_chrom", type=str, default=None, help="Chromosome to use for validation set (e.g., '1')")
    parser.add_argument("--test_chrom", type=str, default=None, help="Chromosome to use for test set (e.g., '2')")
    parser.add_argument("--valid_size", default=0.1, type=float, help="Fraction of data to use for validation if not using chromosome-based split")
    parser.add_argument("--test_size", default=0.1, type=float, help="Fraction of data to use for testing if not using chromosome-based split")
    parser.add_argument("--random_seed", default=42, type=int, help="Random seed for splitting data")
    
    # Filtering arguments
    parser.add_argument("--filter_ids", type=str, help="Optional txt file containing IDs to filter (one ID per line)")
    parser.add_argument("--snp_id_column", type=str, default="SNP_ID", help="Column name containing SNP IDs in format 'chrom_pos_ref_alt'")
    parser.add_argument("-fdh", "--filter_distance_threshold", type=int, default=None, help="Distance threshold for filtering SNPs")
    parser.add_argument("-dc", "--distance_column", type=str, default="abs", help="Column name containing distance to ATG site")

    # Slope normalization arguments
    parser.add_argument("--normalize_slope", action="store_true", help="Normalize slope values using z-score")
    parser.add_argument("--slope_norm_method", type=str, default="zscore", choices=["zscore", "minmax", "robust", "quantile"], help="Normalization method for slope values")
    parser.add_argument("--slope_clip_percentile", type=float, default=None, help="Clip extreme slope values at this percentile (e.g., 95.0)")
    parser.add_argument("--slope_transform", type=str, default=None, choices=["log", "sqrt", "tanh"], help="Transform slope values before normalization")
    return parser.parse_args()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def dna_to_indices(seq_dna: str) -> np.ndarray:
    """Convert DNA sequence to indices for uint8 encoding."""
    dna_alphabet = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
    seq_dna = seq_dna.upper()
    seq_indices = np.zeros(len(seq_dna), dtype=np.uint8)
    seq_indices.fill(4)  # Default to N (4)
    
    for base, idx in dna_alphabet.items():
        seq_indices[np.array([c == base for c in seq_dna])] = idx
    
    return seq_indices


def extract_sequence(fasta: pyfaidx.Fasta, chrom: str, position: int, seq_length: int, strand: str = '+') -> str:
    """Extract sequence of length seq_length centered at position.
    For negative strand, shift window by +1 so that after reverse complement the ATG center aligns.
    """
    half_len = seq_length // 2
    if strand == '+':
        start = max(0, position - half_len)
        end = min(len(fasta[chrom]), position + half_len)
    else:
        start = max(0, position - half_len + 1)
        end = min(len(fasta[chrom]), position + half_len + 1)

    seq = str(fasta[chrom][start:end])
    # Pad with N's if necessary
    if len(seq) < seq_length:
        print(f"Warning: SNP: chr - {chrom}, atg_position: {position+1}, length {len(seq)} is less than {seq_length}, discard this SNP", file=sys.stderr)
        return None
        pad_left = max(0, half_len - position)
        pad_right = max(0, seq_length - len(seq) - pad_left)
        seq = 'N' * pad_left + seq + 'N' * pad_right
    
    # Center the sequence if too long
    if len(seq) > seq_length:
        raise ValueError(f"Sequence length {len(seq)} is greater than {seq_length}, please check the sequence length")
    
    return seq


def reverse_complement(seq: str) -> str:
    """Reverse complement a DNA sequence."""
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
    return ''.join(complement[base] for base in reversed(seq))


def validate_atg_center(seq: str, seq_length: int = 32768, strand: str = '+') -> bool:
    """Validate that the center position contains ATG (positions seq_length//2 : +3)."""
    center_start = seq_length // 2
    center_end = center_start + 3
    center_seq = seq[center_start:center_end].upper()
    return center_seq == "ATG"

def feature_bytes(values: np.ndarray) -> tf.train.Feature:
    """Convert numpy arrays to bytes features."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values.tobytes()]))


def feature_float(value: float) -> tf.train.Feature:
    """Convert a float value to float feature."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def process_batch(job_params: Dict[str, Any]) -> bool:
    """Process a single batch of SNP sequences and write to TFRecord."""
    try:
        tfr_file = job_params['tfr_file']
        batch_data = job_params['batch_data']
        tf_opts = tf.io.TFRecordOptions(compression_type="ZLIB")
        successful_count = 0
        
        with tf.io.TFRecordWriter(tfr_file, tf_opts) as writer:
            for snp_data in batch_data:
                try:
                    ref_seq = snp_data['ref_seq']
                    alt_seq = snp_data['alt_seq']
                    slope = snp_data['slope']
                    
                    # Convert DNA to indices
                    ref_indices = dna_to_indices(ref_seq)
                    alt_indices = dna_to_indices(alt_seq)
                    slope_value = np.float32(slope)
                    
                    # Create features
                    features_dict = {
                        "ref_sequence": feature_bytes(ref_indices),
                        "alt_sequence": feature_bytes(alt_indices),
                        "slope": feature_float(slope_value),
                    }
                    
                    example = tf.train.Example(features=tf.train.Features(feature=features_dict))
                    writer.write(example.SerializeToString())
                    successful_count += 1
                    
                except Exception as e:
                    print(f"Error processing sequence: {e}")
                    continue
        
        return successful_count > 0
        
    except Exception as e:
        print(f"Error processing batch {job_params['tfr_file']}: {e}")
        return False


def find_matching_chromosome(fasta: pyfaidx.Fasta, chrom: str) -> Optional[str]:
    """Find matching chromosome name in FASTA file."""
    candidates = [chrom, f"Chr{chrom}", f"chromosome{chrom}", f"chr{chrom}"]
    for candidate in candidates:
        if candidate in fasta:
            return candidate
    return None


def normalize_slope_values(df: pd.DataFrame, args: argparse.Namespace) -> Tuple[pd.DataFrame, Dict]:
    """Normalize slope values and save normalization parameters.
    
    Args:
        df: DataFrame containing slope data.
        args: Command line arguments.
        
    Returns:
        Tuple[pd.DataFrame, Dict]: Normalized DataFrame and normalization parameters.
    """
    if not args.normalize_slope:
        return df, {}
    
    slope_col = 'Slope'
    original_slopes = df[slope_col].copy()
    
    print(f"Original slope statistics:")
    print(f"  Mean: {original_slopes.mean():.6f}")
    print(f"  Std: {original_slopes.std():.6f}")
    print(f"  Min: {original_slopes.min():.6f}")
    print(f"  Max: {original_slopes.max():.6f}")
    
    # Step 1: Optional clipping of extreme values
    if args.slope_clip_percentile is not None:
        lower_percentile = (100 - args.slope_clip_percentile) / 2
        upper_percentile = 100 - lower_percentile
        
        lower_bound = np.percentile(original_slopes, lower_percentile)
        upper_bound = np.percentile(original_slopes, upper_percentile)
        
        before_clip = len(df)
        df = df[(df[slope_col] >= lower_bound) & (df[slope_col] <= upper_bound)]
        after_clip = len(df)
        
        print(f"Clipped {before_clip - after_clip} extreme values ({args.slope_clip_percentile}% percentile)")
        print(f"  Clip range: [{lower_bound:.6f}, {upper_bound:.6f}]")
        
        # Update slopes after clipping
        slopes = df[slope_col].copy()
    else:
        slopes = original_slopes.copy()
    
    # Step 2: Optional transformation
    if args.slope_transform == "log":
        # Handle negative values by shifting
        min_val = slopes.min()
        if min_val <= 0:
            shift = abs(min_val) + 1e-6
            slopes = slopes + shift
            print(f"Applied log transform with shift: +{shift:.6f}")
        slopes = np.log(slopes)
        transform_params = {"method": "log", "shift": shift if min_val <= 0 else 0}
    elif args.slope_transform == "sqrt":
        # Handle negative values by taking sign and sqrt of absolute value
        slopes = np.sign(slopes) * np.sqrt(np.abs(slopes))
        print("Applied signed square root transform")
        transform_params = {"method": "sqrt"}
    elif args.slope_transform == "tanh":
        slopes = np.tanh(slopes)
        print("Applied tanh transform")
        transform_params = {"method": "tanh"}
    else:
        transform_params = {"method": "none"}
    
    # Step 3: Normalization
    if args.slope_norm_method == "zscore":
        # Z-score normalization (mean=0, std=1)
        mean_val = slopes.mean()
        std_val = slopes.std()
        slopes_norm = (slopes - mean_val) / std_val
        norm_params = {"method": "zscore", "mean": mean_val, "std": std_val}
        print(f"Applied z-score normalization (mean={mean_val:.6f}, std={std_val:.6f})")
        
    elif args.slope_norm_method == "minmax":
        # Min-max normalization (0 to 1)
        min_val = slopes.min()
        max_val = slopes.max()
        slopes_norm = (slopes - min_val) / (max_val - min_val)
        norm_params = {"method": "minmax", "min": min_val, "max": max_val}
        print(f"Applied min-max normalization (min={min_val:.6f}, max={max_val:.6f})")
        
    elif args.slope_norm_method == "robust":
        # Robust normalization using median and IQR
        median_val = slopes.median()
        q75 = slopes.quantile(0.75)
        q25 = slopes.quantile(0.25)
        iqr = q75 - q25
        slopes_norm = (slopes - median_val) / iqr
        norm_params = {"method": "robust", "median": median_val, "iqr": iqr, "q25": q25, "q75": q75}
        print(f"Applied robust normalization (median={median_val:.6f}, IQR={iqr:.6f})")
        
    elif args.slope_norm_method == "quantile":
        # Quantile-based normalization (to uniform distribution)
        from scipy.stats import rankdata
        ranks = rankdata(slopes, method='average')
        slopes_norm = (ranks - 1) / (len(ranks) - 1)  # Scale to [0, 1]
        slopes_norm = slopes_norm * 2 - 1  # Scale to [-1, 1]
        norm_params = {"method": "quantile", "n_samples": len(slopes)}
        print("Applied quantile normalization")
    
    # Update the DataFrame
    df[slope_col] = slopes_norm
    
    print(f"Normalized slope statistics:")
    print(f"  Mean: {slopes_norm.mean():.6f}")
    print(f"  Std: {slopes_norm.std():.6f}")
    print(f"  Min: {slopes_norm.min():.6f}")
    print(f"  Max: {slopes_norm.max():.6f}")
    
    # Combine all parameters
    all_params = {
        "normalization": norm_params,
        "transformation": transform_params,
        "original_stats": {
            "mean": original_slopes.mean(),
            "std": original_slopes.std(),
            "min": original_slopes.min(),
            "max": original_slopes.max()
        }
    }
    
    if args.slope_clip_percentile is not None:
        all_params["clipping"] = {
            "percentile": args.slope_clip_percentile,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound
        }
    
    return df, all_params


def process_single_snp(row: pd.Series, fasta: pyfaidx.Fasta, args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    """Process a single SNP row and return processed data."""
    chrom = str(row['Chrom'])
    fasta_chrom = find_matching_chromosome(fasta, chrom)
    
    if not fasta_chrom:
        raise ValueError(f"Chromosome '{chrom}' not found in FASTA file")
    
    # Get strand information from Direct column
    strand = row['Direct'] if 'Direct' in row else '+'
    if pd.isna(strand) or strand not in ['+', '-']:
        raise ValueError(f"Invalid strand value '{strand}' for {chrom}")
    
    # Parse positions (convert from 1-based to 0-based)
    atg_position_1based = int(row['ATGsite'])
    atg_position_0based = atg_position_1based - 1
    
    # Extract SNP information from SNP_ID
    if args.snp_id_column not in row or pd.isna(row[args.snp_id_column]):
        raise ValueError(f"SNP ID column '{args.snp_id_column}' missing or empty")
    
    snp_id = row[args.snp_id_column]
    parts = snp_id.split('_')
    if len(parts) < 4:
        raise ValueError(f"Invalid SNP ID format '{snp_id}', expected 'chrom_pos_ref_alt'")
    
    snp_position_1based = int(parts[1])
    snp_position_0based = snp_position_1based - 1
    ref_base = parts[-2].upper()
    alt_base = parts[-1].upper()
    slope = row['Slope']
    
    # Extract reference sequence centered at ATG position
    ref_seq = extract_sequence(fasta, fasta_chrom, atg_position_0based, args.seq_length, strand)
    if ref_seq is None:
        return None
    
    # Calculate SNP relative position in the extracted sequence
    center_idx = args.seq_length // 2
    if strand == '+':
        snp_relative_pos = center_idx + (snp_position_0based - atg_position_0based)
    else:
        snp_relative_pos = center_idx + ((snp_position_0based - atg_position_0based - 1))
    
    if not (0 <= snp_relative_pos < len(ref_seq)):
        raise ValueError(f"SNP position {snp_position_1based} outside sequence range for ATG at {atg_position_1based}")
    
    # Validate reference base and create alternate sequence
    actual_ref_base = ref_seq[snp_relative_pos].upper()
    if actual_ref_base != ref_base.upper():
        raise ValueError(f"Reference base mismatch at {chrom}:{snp_position_1based}: expected {ref_base}, found {actual_ref_base}")
    
    alt_seq = ref_seq[:snp_relative_pos] + alt_base + ref_seq[snp_relative_pos + 1:]

    # Apply reverse complement at the very end if negative strand
    if strand == '-':
        ref_seq = reverse_complement(ref_seq)
        alt_seq = reverse_complement(alt_seq)

    # Validate that center position contains ATG after strand correction
    if not validate_atg_center(ref_seq, args.seq_length, strand):
        center_start = args.seq_length // 2
        center_end = center_start + 3
        center_seq = ref_seq[center_start:center_end].upper()
        print(f"Warning: The ATG position({chrom}:{atg_position_0based+1}) of SNP {snp_id} does not contain ATG, found '{center_seq}' instead (strand: {strand})", file=sys.stderr)
        return None
    
    if args.save_fasta:
        fasta_file = os.path.join(args.output_dir, "fasta", f"{chrom}_{snp_position_1based}_{ref_base}_{alt_base}.fasta")
        with open(fasta_file, "a") as f:
            f.write(f">{chrom}_{snp_position_1based}_{ref_base}\n{ref_seq}\n")
            f.write(f">{chrom}_{snp_position_1based}_{alt_base}\n{alt_seq}\n")

    return {
        'ref_seq': ref_seq,
        'alt_seq': alt_seq,
        'slope': slope
    }


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main() -> None:
    args = parse_args()
    
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    else:
        raise ValueError(f"Output directory {args.output_dir} already exists, please delete it or use a different output directory")
    tfr_dir = os.path.join(args.output_dir, "tfrecords")
    if not os.path.isdir(tfr_dir):
        os.makedirs(tfr_dir)
    
    # Load SNP data
    print(f"Loading SNP data from {args.snp_file}...")
    if args.snp_file.endswith('.xlsx'):
        df = pd.read_excel(args.snp_file)
    else:
        separator = '\t' if args.snp_file.endswith('.tsv') else ','
        df = pd.read_csv(args.snp_file, sep=separator)
    print(f"Loaded {len(df)} SNPs")
    
    # Apply filtering if specified
    if args.filter_ids and os.path.exists(args.filter_ids):
        with open(args.filter_ids, 'r') as f:
            filter_ids_set = set(line.strip() for line in f if line.strip())
        print(f"Loaded {len(filter_ids_set)} filter IDs from {args.filter_ids}")
        
        if 'ID' in df.columns:
            df = df[df['ID'].astype(str).isin(filter_ids_set)]
            print(f"After filtering IDs: {len(df)} SNPs remain")
    
    if args.filter_distance_threshold is not None:
        df = df[df[args.distance_column] <= args.filter_distance_threshold]
        print(f"After filtering distance: {len(df)} SNPs remain")
    
    # Normalize slope values if requested
    df, norm_params = normalize_slope_values(df, args)
    
    if args.save_fasta:
        fasta_dir = os.path.join(args.output_dir, "fasta")
        if not os.path.isdir(fasta_dir):
            os.makedirs(fasta_dir)
    
    fasta = pyfaidx.Fasta(args.fasta_file)

    mismatch_count = 0
    for _, row in df.iterrows():
        chrom = str(row['Chrom'])
        fasta_chrom = find_matching_chromosome(fasta, chrom)
            
        if fasta_chrom and args.snp_id_column in row and pd.notna(row[args.snp_id_column]):
            snp_id = row[args.snp_id_column]
            parts = snp_id.split('_')
            if len(parts) >= 4:
                snp_pos = int(parts[1]) - 1
                expected_ref = parts[-2].upper()
                actual_ref = str(fasta[fasta_chrom][snp_pos]).upper()
                if actual_ref != expected_ref:
                   mismatch_count += 1
    if mismatch_count != 0:
        raise ValueError(f"High genome mismatch rate ({mismatch_count}/{len(df)}). Please check genome version compatibility.")
    else:
        print(f"✓ Reference genome matched all SNPs' reference bases")

    # Split data
    if args.valid_chrom is not None and args.test_chrom is not None:
        print(f"Using chromosome-based split: valid={args.valid_chrom}, test={args.test_chrom}")
        valid_chrom_candidates = [args.valid_chrom, int(args.valid_chrom)]
        test_chrom_candidates = [args.test_chrom, int(args.test_chrom)]
        
        valid_mask = df['Chrom'].isin(valid_chrom_candidates)
        test_mask = df['Chrom'].isin(test_chrom_candidates)
        train_mask = ~(valid_mask | test_mask)
        
        train_df = df[train_mask]
        valid_df = df[valid_mask]
        test_df = df[test_mask]
    else:
        print(f"Using random split (valid: {args.valid_size:.1%}, test: {args.test_size:.1%})")
        train_df, temp_df = train_test_split(df, test_size=args.valid_size + args.test_size, random_state=args.random_seed)
        valid_size_adjusted = args.valid_size / (args.valid_size + args.test_size)
        valid_df, test_df = train_test_split(temp_df, test_size=1 - valid_size_adjusted, random_state=args.random_seed)
    
    print(f"Split sizes: Train={len(train_df)}, Valid={len(valid_df)}, Test={len(test_df)}")
    
    # Process SNPs and prepare batch jobs
    print("Preparing batch jobs...")
    all_batch_jobs = []
    
    for split_name, split_df in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
        snp_data_list = []
        batch_counter = 0
        error_snp_count = 0
        for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"Preparing {split_name} data"):
            try:
                processed_data = process_single_snp(row, fasta, args)
                if processed_data:
                    snp_data_list.append(processed_data)
                    
                    if len(snp_data_list) >= args.seqs_per_tfr:
                        job_params = {
                            'tfr_file': os.path.join(tfr_dir, f"{split_name}-{batch_counter}.tfr"),
                            'batch_data': snp_data_list,
                        }
                        all_batch_jobs.append(job_params)
                        snp_data_list = []
                        batch_counter += 1
            except Exception as e:
                chrom = row.get('Chrom', 'unknown')
                atg_pos = row.get('ATGsite', 'unknown')
                error_snp_count += 1
                print(f"Error processing SNP {chrom}:{atg_pos}: {e}, discarding this SNP", file=sys.stderr)
                continue
        
        # Handle remaining data
        if snp_data_list:
            job_params = {
                'tfr_file': os.path.join(tfr_dir, f"{split_name}-{batch_counter}.tfr"),
                'batch_data': snp_data_list,
            }
            all_batch_jobs.append(job_params)
    
    print(f"Created {len(all_batch_jobs)} batch jobs")
    
    # Process batches in parallel
    if args.processes is None:
        args.processes = min(mp.cpu_count(), len(all_batch_jobs))
    
    print(f"Processing batches using {args.processes} processes...")
    with mp.Pool(processes=args.processes) as pool:
        results = list(tqdm(pool.imap_unordered(process_batch, all_batch_jobs), total=len(all_batch_jobs), desc="Processing batches"))
    
    successful = sum(results)
    print(f"✓ {successful}/{len(all_batch_jobs)} batches processed successfully")
    
    # Save statistics
    stats_dict = {
        'train_seqs': len(train_df),
        'valid_seqs': len(valid_df),
        'test_seqs': len(test_df),
        'seq_length': args.seq_length,
    }
    
    
    stats_file = os.path.join(args.output_dir, "statistics.yaml")
    with open(stats_file, "w") as f:
        yaml.dump(stats_dict, f, default_flow_style=False, sort_keys=False, indent=2)
    print(f"{error_snp_count} SNPs are discarded due to errors", file=sys.stderr)
    print("✓ Data processing completed")
    print(f"Final dataset: {len(train_df)} train, {len(valid_df)} valid, {len(test_df)} test sequences")


if __name__ == "__main__":
    main() 