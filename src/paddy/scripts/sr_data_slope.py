#!/usr/bin/env python3
"""
Generate TFRecord files for SloRice regression model from SNP data.

This script processes SNP data and generates TFRecord files containing:
- ref_sequence: Reference DNA sequence (reverse complemented if negative strand)
- alt_sequence: Alternate DNA sequence with SNP (reverse complemented if negative strand)
- slope: continuous label from 'Slope' column
- strand: Strand information (0: positive, 1: negative)

The script handles strand information from the 'Direct' column. For negative strand,
reverse complement is applied at the final step after all validations, ensuring
the center position (16385:16387) contains ATG in the final output.

The script uses parallel processing to generate TFRecord files efficiently,
similar to the classification script but writing continuous labels.
"""

import argparse
import os
import sys
import pickle
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import pyfaidx
import yaml
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Generate TFRecord files for SloRice regression model from SNP data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument('snp_file', type=str, help='Excel/CSV/TSV file containing SNP data')
    parser.add_argument('fasta_file', type=str, help='FASTA file containing reference genome')

    # Optional arguments
    parser.add_argument("-s", "--seqs_per_tfr", default=256, type=int, help="Sequences per TFRecord file")
    parser.add_argument("-o", "--output_dir", default="sr_data_slope_out", help="Output directory")
    parser.add_argument("-p", "--processes", default=None, type=int, help="Number of parallel processes")
    parser.add_argument("--run_local", action="store_true", default=True, help="Run jobs locally (default)")
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


def reverse_complement(seq: str) -> str:
    """Reverse complement a DNA sequence."""
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
    return ''.join(complement[base] for base in reversed(seq))


def extract_sequence(fasta: pyfaidx.Fasta, chrom: str, position: int, seq_length: int, strand: str = '+') -> Optional[str]:
    """Extract sequence of length seq_length centered at position."""
    half_len = seq_length // 2
    if strand == '+':
        start = max(0, position - half_len)
        end = min(len(fasta[chrom]), position + half_len)
    else:
        start = max(0, position - half_len + 1)  # +1 to align ATG after RC
        end = min(len(fasta[chrom]), position + half_len + 1)

    seq = str(fasta[chrom][start:end])
    if len(seq) < seq_length:
        print(f"Warning: SNP: chr - {chrom}, atg_position: {position+1}, length {len(seq)} is less than {seq_length}, discard this SNP", file=sys.stderr)
        return None

    if len(seq) > seq_length:
        raise ValueError(f"Sequence length {len(seq)} is greater than {seq_length}, please check the sequence length")

    return seq


def validate_atg_center(seq: str, seq_length: int = 32768, strand: str = '+') -> bool:
    """Validate that the center position contains ATG."""
    center_start = seq_length // 2
    center_end = center_start + 3
    center_seq = seq[center_start:center_end].upper()
    return center_seq == "ATG"


def create_batch_data_string(batch_data: List[Dict[str, Any]]) -> str:
    """Create a string representation of batch data for command line passing."""
    import base64
    import gzip

    batch_bytes = pickle.dumps(batch_data)
    compressed = gzip.compress(batch_bytes)
    encoded = base64.b64encode(compressed).decode('ascii')
    return encoded


def run_job(cmd):
    """Execute a single job command."""
    try:
        import subprocess
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)


def exec_par(jobs, processes=None, verbose=True):
    """Execute jobs in parallel using multiprocessing."""
    import multiprocessing as mp

    if processes is None:
        processes = min(mp.cpu_count(), len(jobs))

    if verbose:
        print(f"Executing {len(jobs)} jobs using {processes} processes...")

    with mp.Pool(processes=processes) as pool:
        results = list(tqdm(pool.imap_unordered(run_job, jobs), total=len(jobs), desc="Processing batches"))

    successful = sum(1 for success, _, _ in results if success)
    failed = len(jobs) - successful

    if verbose:
        print(f"Completed: {successful} successful, {failed} failed")
        for i, (success, stdout, stderr) in enumerate(results):
            if not success:
                print(f"Job {i} failed:")
                if stderr:
                    print(f"  Error: {stderr}")
                if stdout:
                    print(f"  Output: {stdout}")

    return successful, failed


def find_matching_chromosome(fasta: pyfaidx.Fasta, chrom: str) -> Optional[str]:
    """Find matching chromosome name in FASTA file."""
    candidates = [chrom, f"Chr{chrom}", f"chromosome{chrom}", f"chr{chrom}"]
    for candidate in candidates:
        if candidate in fasta:
            return candidate
    return None


def process_single_snp(row: pd.Series, fasta: pyfaidx.Fasta, args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    """Process a single SNP row and return processed data for regression."""
    chrom = str(row['Chrom'])
    fasta_chrom = find_matching_chromosome(fasta, chrom)

    if not fasta_chrom:
        raise ValueError(f"Chromosome '{chrom}' not found in FASTA file")

    strand = row['Direct']
    if pd.isna(strand) or strand not in ['+', '-']:
        raise ValueError(f"Invalid strand value '{strand}' for {chrom}")

    atg_position_0based = int(row['ATGsite']) - 1

    if args.snp_id_column not in row or pd.isna(row[args.snp_id_column]):
        raise ValueError(f"SNP ID column '{args.snp_id_column}' missing or empty")

    snp_id = row[args.snp_id_column]
    parts = snp_id.split('_')
    if len(parts) < 4:
        raise ValueError(f"Invalid SNP ID format '{snp_id}', expected 'chrom_pos_ref_alt'")

    snp_position_0based = int(parts[1]) - 1
    ref_base = parts[-2].upper()
    alt_base = parts[-1].upper()
    slope_value = float(row['Slope'])

    ref_seq = extract_sequence(fasta, fasta_chrom, atg_position_0based, args.seq_length, strand)
    if ref_seq is None:
        return None

    center_idx = args.seq_length // 2
    if strand == '+':
        snp_relative_pos = center_idx + (snp_position_0based - atg_position_0based)
    else:
        snp_relative_pos = center_idx + ((snp_position_0based - atg_position_0based - 1))

    actual_ref_base = ref_seq[snp_relative_pos].upper()
    if actual_ref_base != ref_base.upper():
        raise ValueError(f"Reference base mismatch at {chrom}:{snp_position_0based}(0-based): expected {ref_base}, found {actual_ref_base}")

    alt_seq = ref_seq[:snp_relative_pos] + alt_base + ref_seq[snp_relative_pos + 1:]

    if strand == '-':
        ref_seq = reverse_complement(ref_seq)
        alt_seq = reverse_complement(alt_seq)

    if not validate_atg_center(ref_seq, args.seq_length, strand):
        center_start = args.seq_length // 2
        center_end = center_start + 3
        center_seq = ref_seq[center_start:center_end].upper()
        print(f"Warning: The ATG position({chrom}:{atg_position_0based+1}) of SNP {snp_id} does not contain ATG, found '{center_seq}' instead (strand: {strand})", file=sys.stderr)
        return None

    if args.save_fasta:
        snp_position_1based = snp_position_0based + 1
        fasta_file = os.path.join(args.output_dir, "fasta", f"{chrom}_{snp_position_1based}_{ref_base}_{alt_base}.fasta")
        with open(fasta_file, "a") as f:
            f.write(f">{chrom}_{snp_position_1based}_{ref_base}\n{ref_seq}\n")
            f.write(f">{chrom}_{snp_position_1based}_{alt_base}\n{alt_seq}\n")

    return {
        'ref_seq': ref_seq,
        'alt_seq': alt_seq,
        'slope': slope_value,
        'strand': strand
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
    batch_dir = os.path.join(args.output_dir, "batches")
    if not os.path.isdir(batch_dir):
        os.makedirs(batch_dir)
    if args.save_fasta:
        fasta_dir = os.path.join(args.output_dir, "fasta")
        if not os.path.isdir(fasta_dir):
            os.makedirs(fasta_dir)

    print(f"Loading SNP data from {args.snp_file}...")
    if args.snp_file.endswith('.xlsx'):
        df = pd.read_excel(args.snp_file)
    else:
        separator = '\t' if args.snp_file.endswith('.tsv') else ','
        df = pd.read_csv(args.snp_file, sep=separator)
    print(f"Loaded {len(df)} SNPs")

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

    fasta = pyfaidx.Fasta(args.fasta_file)
    mismatch_count = 0
    for _, row in df.iterrows():
        chrom = str(row['Chrom'])

        if chrom and args.snp_id_column in row and pd.notna(row[args.snp_id_column]):
            snp_id = row[args.snp_id_column]
            parts = snp_id.split('_')
            if len(parts) >= 4:
                snp_pos = int(parts[1]) - 1
                expected_ref = parts[-2].upper()
                actual_ref = str(fasta[chrom][snp_pos]).upper()
                if actual_ref != expected_ref:
                   mismatch_count += 1
    if mismatch_count != 0:
        raise ValueError(f"Genome mismatch counts: {mismatch_count}. Please check genome compatibility.")
    else:
        print(f"\u2713 Reference genome matched all SNPs' reference bases")

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

    print("Preparing batch commands...")

    job_commands = []
    total_error_count = 0

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
                        batch_data_encoded = create_batch_data_string(snp_data_list)
                        batch_file = os.path.join(batch_dir, f"{split_name}-{batch_counter}.b64")
                        with open(batch_file, "w") as f:
                            f.write(batch_data_encoded)
                        tfr_file = os.path.join(tfr_dir, f"{split_name}-{batch_counter}.tfr")
                        cmd = f"sr_data_slope_write.py {batch_file} {tfr_file}"
                        job_commands.append(cmd)

                        snp_data_list = []
                        batch_counter += 1
                else:
                    error_snp_count += 1

            except Exception as e:
                chrom = row.get('Chrom', 'unknown')
                atg_pos = row.get('ATGsite', 'unknown')
                error_snp_count += 1
                print(f"Error processing SNP {chrom}:{atg_pos}: {e}", file=sys.stderr)

        if snp_data_list:
            batch_data_encoded = create_batch_data_string(snp_data_list)
            batch_file = os.path.join(batch_dir, f"{split_name}-{batch_counter}.b64")
            with open(batch_file, "w") as f:
                f.write(batch_data_encoded)
            tfr_file = os.path.join(tfr_dir, f"{split_name}-{batch_counter}.tfr")
            cmd = f"sr_data_slope_write.py {batch_file} {tfr_file}"
            job_commands.append(cmd)

        if error_snp_count > 0:
            print(f"{error_snp_count} SNPs in {split_name} split were discarded due to errors", file=sys.stderr)
            total_error_count += error_snp_count

    print(f"Created {len(job_commands)} batch commands")

    successful, failed = exec_par(job_commands, args.processes, verbose=True)

    if failed > 0:
        print(f"Warning: {failed} batches failed", file=sys.stderr)

    print(f"\u2713 {successful}/{len(job_commands)} batches processed successfully")

    stats_dict = {
        'train_seqs': len(train_df),
        'valid_seqs': len(valid_df),
        'test_seqs': len(test_df),
        'seq_length': args.seq_length,
        'label': 'slope',
    }

    if 'Direct' in df.columns:
        strand_counts = df['Direct'].value_counts()
        stats_dict['strand_distribution'] = {
            'positive_strand': int(strand_counts.get('+', 0)),
            'negative_strand': int(strand_counts.get('-', 0)),
            'unknown_strand': int(strand_counts.sum() - strand_counts.get('+', 0) - strand_counts.get('-', 0))
        }
        print(f"Strand distribution:")
        print(f"  Positive strand (+): {strand_counts.get('+', 0)} SNPs")
        print(f"  Negative strand (-): {strand_counts.get('-', 0)} SNPs")
        if strand_counts.sum() - strand_counts.get('+', 0) - strand_counts.get('-', 0) > 0:
            print(f"  Unknown strand: {strand_counts.sum() - strand_counts.get('+', 0) - strand_counts.get('-', 0)} SNPs")

    stats_file = os.path.join(args.output_dir, "statistics.yaml")
    with open(stats_file, "w") as f:
        yaml.dump(stats_dict, f, default_flow_style=False, sort_keys=False, indent=2)

    if total_error_count > 0:
        print(f"{total_error_count} SNPs were discarded due to errors", file=sys.stderr)

    print("\u2713 Data processing completed")
    print(f"Final dataset: {len(train_df)} train, {len(valid_df)} valid, {len(test_df)} test sequences")
    print(f"Statistics saved to: {stats_file}")


if __name__ == "__main__":
    main()


