#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import sys
import multiprocessing as mp
import yaml
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from collections import defaultdict
from paddy.utils import legalize_identifier
"""
Generate TFRecord files for seq2exp model directly from TSV file.
"""

def parse_args():
    parser = argparse.ArgumentParser(description='Generate TFRecord files from TSV file')
    parser.add_argument(
        'tsv_file',
        type=str,
        help='TSV file containing sequences and expression data'
    )
    parser.add_argument(
        "-s",
        "--seqs_per_tfr",
        default=512,
        type=int,
        help="Sequences per TFRecord file [Default: %(default)s]",
    )
    parser.add_argument(
        "-d",
        "--decimals",
        default=None,
        type=int,
        help="Round values to given decimals [Default: %(default)s]",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        default="out_data_transfer",
        help="Output directory [Default: %(default)s]",
    )
    parser.add_argument(
        "-p",
        "--processes",
        default=None,
        type=int,
        help="Number of parallel processes [Default: %(default)s]",
    )
    parser.add_argument(
        "-l",
        "--seq_length",
        default=32768,
        type=int,
        help="Sequence length [Default: %(default)s]",
    )
    parser.add_argument(
        "-w",
        "--pool_width",
        default=32,
        type=int,
        help="Sum pool width [Default: %(default)s]",
    )
    parser.add_argument(
        "-c",
        "--crop_bp",
        default=0,
        type=int,
        help="Crop bp off each end [Default: %(default)s]",
    )
    parser.add_argument(
        "-t",
        "--targets_file",
        help="Targets file to copy (optional)",
    )
    parser.add_argument(
        "--valid_chrom",
        default=None,
        help="Chromosome to use for validation set (e.g., 'Chr1')",
    )
    parser.add_argument(
        "--test_chrom",
        default=None,
        help="Chromosome to use for test set (e.g., 'Chr2')",
    )
    parser.add_argument(
        "--chrom_column",
        default=2,
        type=int,
        help="0-based index of chromosome column in TSV file [Default: %(default)s]",
    )
    parser.add_argument(
        "--seq_column",
        default=5,
        type=int,
        help="0-based index of DNA sequence column in TSV file [Default: %(default)s]",
    )
    parser.add_argument(
        "--identifier_column",
        default=0,
        type=int,
        help="0-based index of identifier column in TSV file [Default: %(default)s]",
    )
    parser.add_argument(
        "--target_start_column",
        default=6,
        type=int,
        help="0-based index of first target column in TSV file [Default: %(default)s]",
    )
    parser.add_argument(
        "--valid_size",
        default=0.1,
        type=float,
        help="Fraction of data to use for validation if not using chromosome-based split [Default: %(default)s]",
    )
    parser.add_argument(
        "--test_size",
        default=0.1,
        type=float,
        help="Fraction of data to use for testing if not using chromosome-based split [Default: %(default)s]",
    )
    parser.add_argument(
        "--random_seed",
        default=42,
        type=int,
        help="Random seed for splitting data [Default: %(default)s]",
    )
    parser.add_argument(
        "--chunk_size",
        default=10000,
        type=int,
        help="Number of rows to read at once from TSV file [Default: %(default)s]",
    )
    parser.add_argument(
        "--filter_ids",
        type=str,
        help="Optional txt file containing IDs to filter (one ID per line). Only rows with matching first column ID will be processed.",
    )
    return parser.parse_args()


def dna_to_indices(seq_dna):
    """Convert DNA sequence to indices for uint8 encoding."""
    # Define DNA alphabet mapping
    dna_alphabet = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
    
    # Convert to uppercase once
    seq_dna = seq_dna.upper()
    
    # Use numpy's vectorized approach for better performance
    seq_indices = np.zeros(len(seq_dna), dtype=np.uint8)
    seq_indices.fill(4)  # Default to N (4)
    
    # Set values for each base (faster than looping through each character)
    for base, idx in dna_alphabet.items():
        seq_indices[np.array([c == base for c in seq_dna])] = idx
    
    return seq_indices


def feature_bytes(values):
    """Convert numpy arrays to bytes features."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values.tobytes()]))


def process_batch(job_params):
    """Process a single batch of sequences and write to TFRecord."""
    try:
        # Extract parameters
        tfr_file = job_params['tfr_file']
        batch_indices = job_params['batch_indices']
        sequences = job_params['sequences']
        targets = job_params['targets']
        decimals = job_params['decimals']
        seq_length = job_params['seq_length']
        
        # Define TFRecord options
        tf_opts = tf.io.TFRecordOptions(compression_type="ZLIB")
        
        # Record the number of sequences successfully processed
        successful_count = 0
        
        with tf.io.TFRecordWriter(tfr_file, tf_opts) as writer:
            for idx in batch_indices:
                try:
                    seq_dna = sequences[idx]
                    target_values = targets[idx].copy()  # Create a copy to avoid modifying the original data
                    
                    # Validate sequence length
                    if len(seq_dna) == 0:
                        continue
                    
                    # Ensure sequence length matches expected length
                    if len(seq_dna) != seq_length:
                        if len(seq_dna) > seq_length:
                            # Truncate from the center to preserve both ends
                            excess = len(seq_dna) - seq_length
                            start_cut = excess // 2
                            seq_dna = seq_dna[start_cut:start_cut+seq_length]
                        else:
                            # Pad with N's to reach required length
                            padding_needed = seq_length - len(seq_dna)
                            # Add padding equally to both sides
                            left_pad = padding_needed // 2
                            right_pad = padding_needed - left_pad
                            seq_dna = 'N' * left_pad + seq_dna + 'N' * right_pad
                    
                    # Convert DNA to indices (uint8 encoding, compatible with paddy)
                    seq_indices = dna_to_indices(seq_dna)
                    
                    # Process targets
                    if decimals is not None:
                        target_values = np.around(target_values, decimals=decimals)
                    
                    # Ensure no infinite values and convert to float32
                    target_values = np.nan_to_num(target_values, nan=0.0, posinf=0.0, neginf=0.0)
                    target_values = target_values.astype(np.float32)
                    
                    # Validate data integrity
                    if seq_indices.size == 0 or target_values.size == 0:
                        continue
                    
                    # Create features dictionary (compatible with paddy format)
                    features_dict = {
                        "sequence": feature_bytes(seq_indices),  # uint8 encoded DNA sequence
                        "target": feature_bytes(target_values),  # float32 targets
                    }
                    
                    # Write example
                    example = tf.train.Example(
                        features=tf.train.Features(feature=features_dict)
                    )
                    writer.write(example.SerializeToString())
                    successful_count += 1
                    
                except Exception as e:
                    print(f"Error processing sequence {idx}: {e}")
                    continue
        
        if successful_count == 0:
            print(f"Warning: No sequences successfully processed for {tfr_file}")
            return False
        
        return True
        
    except Exception as e:
        print(f"Error processing batch {tfr_file}: {e}")
        return False


def count_tsv_rows(file_path):
    """Count the number of rows in a TSV file without loading it into memory."""
    print(f"Counting rows in {file_path}...")
    
    # Use wc -l command for faster counting on large files
    try:
        import subprocess
        result = subprocess.run(['wc', '-l', file_path], capture_output=True, text=True)
        if result.returncode == 0:
            # Extract the number from the output (first field)
            count = int(result.stdout.strip().split()[0])
            # Subtract 1 for the header
            return count - 1
    except:
        pass
    
    # Fallback: count lines manually
    count = 0
    with open(file_path, 'r') as f:
        # Skip header
        next(f)
        for _ in f:
            count += 1
            if count % 1000000 == 0:
                print(f"Counted {count:,} rows...")
    
    return count


def process_tsv_in_chunks(args):
    """Process the TSV file in chunks to avoid loading everything into memory."""
    # Count total rows first (if needed)
    total_rows = count_tsv_rows(args.tsv_file)
    print(f"Total rows in TSV file: {total_rows:,}")

    # Load filter IDs if provided
    filter_ids_set = None
    if args.filter_ids:
        if os.path.exists(args.filter_ids):
            with open(args.filter_ids, 'r') as f:
                filter_ids_set = set(line.strip() for line in f if line.strip())
            print(f"Loaded {len(filter_ids_set)} filter IDs from {args.filter_ids}")
        else:
            print(f"Warning: Filter IDs file '{args.filter_ids}' not found. Proceeding without filtering.")
            filter_ids_set = None

    # Create output directory
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Create TFRecord directory
    tfr_dir = os.path.join(args.output_dir, "tfrecords")
    if not os.path.isdir(tfr_dir):
        os.makedirs(tfr_dir)
    
    # Prepare for chromosome-based or random split
    if args.valid_chrom is not None and args.test_chrom is not None:
        # For chromosome-based split, we need to track indices by chromosome
        train_indices = []
        valid_indices = []
        test_indices = []
        
        # Convert to uppercase for case-insensitive comparison
        valid_chrom_str = str(args.valid_chrom).upper()
        test_chrom_str = str(args.test_chrom).upper()
        
        print(f"Using chromosome-based split:")
        print(f"  - Validation: {args.valid_chrom}")
        print(f"  - Test: {args.test_chrom}")
        print(f"  - Train: all other chromosomes")
    else:
        # For random split, we need to read all indices first
        print(f"Using random split (valid: {args.valid_size:.1%}, test: {args.test_size:.1%})")
        
        # Generate all indices
        all_indices = np.arange(total_rows)
        
        # First split off the test set
        remaining_indices, test_indices = train_test_split(
            all_indices, 
            test_size=args.test_size,
            random_state=args.random_seed
        )
        
        # Then split the remaining data into train and validation
        valid_size_adjusted = args.valid_size / (1 - args.test_size)  # Adjust for the remaining data
        train_indices, valid_indices = train_test_split(
            remaining_indices,
            test_size=valid_size_adjusted,
            random_state=args.random_seed
        )
        
        # Convert to sets for faster lookup
        train_indices_set = set(train_indices)
        valid_indices_set = set(valid_indices)
        test_indices_set = set(test_indices)
    
    # Process in chunks
    chunk_size = args.chunk_size
    num_chunks = (total_rows + chunk_size - 1) // chunk_size
    
    # Initialize counters for statistics
    total_train = 0
    total_valid = 0
    total_test = 0
    num_targets = None
    
    # Prepare batch jobs for parallel processing
    batch_jobs = []
    # Set to collect all unique legalized identifiers
    unique_legal_ids = set()
    
    # Keep track of global batch indices for each flag group
    batch_counters = {'train': 0, 'valid': 0, 'test': 0}
    
    # Create a reader that will read the file in chunks
    reader = pd.read_csv(
        args.tsv_file, 
        sep='\t', 
        chunksize=chunk_size,
        low_memory=False
    )
    
    # Process each chunk
    global_index = 0
    for chunk_idx, chunk in enumerate(tqdm(reader, total=num_chunks, desc="Reading chunks")):
        
        # Extract data from the chunk
        chunk_chromosomes = chunk.iloc[:, args.chrom_column].values
        chunk_sequences = chunk.iloc[:, args.seq_column].values
        chunk_targets = chunk.iloc[:, args.target_start_column:].values
        chunk_identifier = chunk.iloc[:, args.identifier_column].values
        
        # Apply ID filtering if filter_ids_set is provided
        if filter_ids_set is not None:
            # Create mask for rows to keep
            keep_mask = np.array([str(id_val) in filter_ids_set for id_val in chunk_identifier])
            
            # Apply filtering
            chunk_chromosomes = chunk_chromosomes[keep_mask]
            chunk_sequences = chunk_sequences[keep_mask]
            chunk_targets = chunk_targets[keep_mask]
            chunk_identifier = chunk_identifier[keep_mask]
            
            print(f"  After filtering: {len(chunk_identifier)} rows kept from {len(keep_mask)} total")
        
        # Record number of targets on first chunk
        if num_targets is None:
            num_targets = chunk_targets.shape[1]
            print(f"Number of targets: {num_targets}")
        
        # Assign indices to train/valid/test based on the split method
        chunk_train_indices = []
        chunk_valid_indices = []
        chunk_test_indices = []
        
        if args.valid_chrom is not None and args.test_chrom is not None:
            # Chromosome-based split
            for i, chrom in enumerate(chunk_chromosomes):
                # Ensure case-insensitive comparison
                chrom_str = str(chrom).upper()
                
                if chrom_str == valid_chrom_str:
                    chunk_valid_indices.append(i)
                    valid_indices.append(global_index + i)
                elif chrom_str == test_chrom_str:
                    chunk_test_indices.append(i)
                    test_indices.append(global_index + i)
                else:
                    chunk_train_indices.append(i)
                    train_indices.append(global_index + i)
        else:
            # Random split (use pre-computed indices)
            for i in range(len(chunk)):
                idx = global_index + i
                if idx in train_indices_set:
                    chunk_train_indices.append(i)
                elif idx in valid_indices_set:
                    chunk_valid_indices.append(i)
                elif idx in test_indices_set:
                    chunk_test_indices.append(i)
        
        # Update counters
        total_train += len(chunk_train_indices)
        total_valid += len(chunk_valid_indices)
        total_test += len(chunk_test_indices)
        
        # Create batch jobs for this chunk
        chunk_groups = {
            'train': chunk_train_indices,
            'valid': chunk_valid_indices,
            'test': chunk_test_indices
        }
        
        
        
        for flag, indices in chunk_groups.items():
            if not indices:
                continue
            # Group by identifier
            id_to_indices = defaultdict(list)
            for idx in indices:
                id_val = chunk_identifier[idx]
                id_to_indices[id_val].append(idx)
            for id_val, id_indices in id_to_indices.items():
                legal_id = legalize_identifier(id_val)
                unique_legal_ids.add(legal_id)
                batch_size = args.seqs_per_tfr
                num_batches = (len(id_indices) + batch_size - 1) // batch_size
                for batch_idx in range(num_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, len(id_indices))
                    batch_indices = id_indices[start_idx:end_idx]
                    job_params = {
                        'tfr_file': os.path.join(tfr_dir, f"{flag}-{legal_id}-{batch_idx}.tfr"),
                        'batch_indices': batch_indices,
                        'sequences': chunk_sequences,
                        'targets': chunk_targets,
                        'decimals': args.decimals,
                        'seq_length': args.seq_length
                    }
                    batch_jobs.append(job_params)
        # Update global index
        global_index += len(chunk)
    
    # After all batches are processed, write unique legalized identifiers to file
    unique_id_file = os.path.join(args.output_dir, "unique_identifiers.txt")
    with open(unique_id_file, "w") as f:
        for legal_id in sorted(unique_legal_ids):
            f.write(f"{legal_id}\n")
    print(f"Unique legalized identifiers written to: {unique_id_file}")

    # Process batches in parallel
    if args.processes is None:
        args.processes = min(mp.cpu_count(), len(batch_jobs))
    else:
        args.processes = min(args.processes, mp.cpu_count())
    print(f"Processing {len(batch_jobs)} batches using {args.processes} processes...")
    
    with mp.Pool(processes=args.processes) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_batch, batch_jobs),
            total=len(batch_jobs),
            desc="Processing batches"
        ))
    
    # Report results
    successful = sum(results)
    total = len(batch_jobs)
    print(f"TFRecord generation completed! {successful}/{total} batches processed successfully.")
    if total - successful > 0:
        print(f"Warning: {total - successful} batches failed to process.")
    
    # Print final dataset sizes
    print(f"Final dataset sizes:")
    print(f"Train: {total_train} sequences")
    print(f"Valid: {total_valid} sequences")
    print(f"Test: {total_test} sequences")
    
    # Print filtering statistics if filtering was applied
    if filter_ids_set is not None:
        total_processed = total_train + total_valid + total_test
        print(f"\nFiltering statistics:")
        print(f"  Original filter IDs: {len(filter_ids_set)}")
        print(f"  Total sequences processed: {total_processed}")
    
    return {
        'num_targets': num_targets,
        'train_seqs': total_train,
        'valid_seqs': total_valid,
        'test_seqs': total_test,
        'seq_length': args.seq_length,
        'pool_width': args.pool_width,
        'crop_bp': args.crop_bp
    }


def main():
    args = parse_args()
    
    # Process TSV file in chunks
    stats_dict = process_tsv_in_chunks(args)

    ################################################################
    # Generate statistics
    ################################################################
    print("Generating statistics...")
    
    # Add additional statistics
    stats_dict["seq_1hot"] = True
    
    # Write statistics to YAML file
    stats_file = os.path.join(args.output_dir, "statistics.yaml")
    with open(stats_file, "w") as stats_yaml_out:
        yaml.dump(stats_dict, stats_yaml_out, default_flow_style=False, sort_keys=False, indent=2)
    
    print(f"Statistics written to: {stats_file}")
    
    ################################################################
    # Copy targets.txt if provided
    ################################################################
    if args.targets_file:
        print("Copying targets.txt...")
        targets_output_file = os.path.join(args.output_dir, "targets.txt")
        
        if os.path.exists(args.targets_file):
            import shutil
            shutil.copy(args.targets_file, targets_output_file)
            print(f"Copied targets file: {args.targets_file} -> {targets_output_file}")
        else:
            print(f"Warning: Targets file '{args.targets_file}' not found, skipping.")
    
    print("Data processing completed!")


if __name__ == "__main__":
    main() 