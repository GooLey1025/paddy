#! /usr/bin/env python3
from optparse import OptionParser
import pysam
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import sys
import multiprocessing as mp
from functools import partial
import yaml
"""
generate TFRecord files for transfer learning of seq2exp model based on seq2chromatin model
"""

def main():
    usage = "usage: %prog [options] <fasta_file> <xlsx_file>"
    parser = OptionParser(usage)
    parser.add_option(
        "-s",
        dest="seqs_per_tfr",
        default=256,
        type="int",
        help="Sequences per TFRecord file [Default: %default]",
    )
    parser.add_option(
        "-d",
        dest="decimals",
        default=None,
        type="int",
        help="Round values to given decimals [Default: %default]",
    )
    parser.add_option(
        "-o",
        dest="output_dir",
        default="out_data_transfer",
        help="Output directory [Default: %default]",
    )
    parser.add_option(
        "-p",
        dest="processes",
        default=None,
        type="int",
        help="Number of parallel processes [Default: %default]",
    )
    parser.add_option(
        "-l",
        dest="seq_length",
        default=32768,
        type="int",
        help="Sequence length [Default: %default]",
    )
    parser.add_option(
        "-w",
        dest="pool_width",
        default=32,
        type="int",
        help="Sum pool width [Default: %default]",
    )
    parser.add_option(
        "-c",
        "--crop",
        dest="crop_bp",
        default=0,
        type="int",
        help="Crop bp off each end [Default: %default]",
    )
    parser.add_option(
        "-t",
        "--targets",
        dest="targets_file",
        help="Targets file to copy (optional)",
    )
    (options, args) = parser.parse_args()

    if len(args) != 2:
        parser.error("Must provide input arguments.")
    else:
        fasta_file = args[0]
        xlsx_file = args[1]  # contains sequence and targets

    # Create output directory
    if not os.path.isdir(options.output_dir):
        os.makedirs(options.output_dir)

    # Read xlsx file
    print("Reading xlsx file...")
    df = pd.read_excel(xlsx_file)
    
    # Extract chromosome, start, end, flag (columns 5-8, 0-indexed as 4-7)
    # and targets (columns 9 onwards, 0-indexed as 8 onwards)
    chromosomes = df.iloc[:, 4].values  # 5th column
    starts = df.iloc[:, 5].values       # 6th column  
    ends = df.iloc[:, 6].values         # 7th column
    flags = df.iloc[:, 7].values        # 8th column
    targets = df.iloc[:, 8:].values     # 9th column onwards
    
    print(f"Loaded {len(chromosomes)} sequences")
    print(f"Number of targets: {targets.shape[1]}")
    
    # Group sequences by flag (train/valid/test)
    flag_groups = {}
    for i, flag in enumerate(flags):
        if flag not in flag_groups:
            flag_groups[flag] = []
        flag_groups[flag].append(i)
    
    print(f"Found groups: {list(flag_groups.keys())}")
    
    # Create TFRecord directory
    tfr_dir = os.path.join(options.output_dir, "tfrecords")
    if not os.path.isdir(tfr_dir):
        os.makedirs(tfr_dir)
    
    # Prepare batch jobs for parallel processing
    batch_jobs = []
    
    for flag, indices in flag_groups.items():
        print(f"Preparing {flag} group with {len(indices)} sequences...")
        
        # Split into batches
        batch_size = options.seqs_per_tfr
        num_batches = (len(indices) + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(indices))
            batch_indices = indices[start_idx:end_idx]
            
            # Create job parameters
            job_params = {
                'fasta_file': fasta_file,
                'tfr_file': os.path.join(tfr_dir, f"{flag}-{batch_idx}.tfr"),
                'batch_indices': batch_indices,
                'chromosomes': chromosomes,
                'starts': starts,
                'ends': ends,
                'targets': targets,
                'decimals': options.decimals
            }
            batch_jobs.append(job_params)
    
    # Process batches in parallel
    if options.processes is None:
        options.processes = min(mp.cpu_count(), len(batch_jobs))
    
    print(f"Processing {len(batch_jobs)} batches using {options.processes} processes...")
    
    with mp.Pool(processes=options.processes) as pool:
        results = pool.map(process_batch, batch_jobs)
    
    # Report results
    successful = sum(results)
    total = len(batch_jobs)
    print(f"TFRecord generation completed! {successful}/{total} batches processed successfully.")
    
    ################################################################
    # Generate statistics
    ################################################################
    print("Generating statistics...")
    
    # Calculate statistics
    stats_dict = {}
    stats_dict["num_targets"] = targets.shape[1]
    stats_dict["seq_length"] = options.seq_length
    stats_dict["seq_1hot"] = True
    stats_dict["pool_width"] = options.pool_width
    stats_dict["crop_bp"] = options.crop_bp
    
    # Count sequences by flag
    for flag in flag_groups.keys():
        stats_dict[f"{flag}_seqs"] = len(flag_groups[flag])
    
    # Write statistics to YAML file
    stats_file = os.path.join(options.output_dir, "statistics.yaml")
    with open(stats_file, "w") as stats_yaml_out:
        yaml.dump(stats_dict, stats_yaml_out, default_flow_style=False, sort_keys=False, indent=2)
    
    print(f"Statistics written to: {stats_file}")
    
    ################################################################
    # Copy targets.txt if provided
    ################################################################
    if options.targets_file:
        print("Copying targets.txt...")
        targets_output_file = os.path.join(options.output_dir, "targets.txt")
        
        if os.path.exists(options.targets_file):
            import shutil
            shutil.copy(options.targets_file, targets_output_file)
            print(f"Copied targets file: {options.targets_file} -> {targets_output_file}")
        else:
            print(f"Warning: Targets file '{options.targets_file}' not found, skipping.")
    
    print("Data transfer processing completed!")


def dna_to_indices(seq_dna):
    """Convert DNA sequence to indices for uint8 encoding."""
    # Define DNA alphabet mapping
    dna_alphabet = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
    
    # Convert to indices
    seq_indices = []
    for base in seq_dna.upper():
        seq_indices.append(dna_alphabet.get(base, 4))  # N for unknown bases
    
    return np.array(seq_indices, dtype=np.uint8)


def feature_bytes(values):
    """Convert numpy arrays to bytes features."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values.tobytes()]))


def process_batch(job_params):
    """Process a single batch of sequences and write to TFRecord."""
    fasta_open = None
    writer = None
    
    try:
        # Extract parameters
        fasta_file = job_params['fasta_file']
        tfr_file = job_params['tfr_file']
        batch_indices = job_params['batch_indices']
        chromosomes = job_params['chromosomes']
        starts = job_params['starts']
        ends = job_params['ends']
        targets = job_params['targets']
        decimals = job_params['decimals']
        
        # Open FASTA file for this process
        fasta_open = pysam.Fastafile(fasta_file)
        
        # Define TFRecord options
        tf_opts = tf.io.TFRecordOptions(compression_type="ZLIB")
        
        # 记录成功处理的数量
        successful_count = 0
        
        with tf.io.TFRecordWriter(tfr_file, tf_opts) as writer:
            for idx in batch_indices:
                try:
                    chr_name = str(chromosomes[idx])
                    start = int(starts[idx])
                    end = int(ends[idx])
                    target_values = targets[idx].copy()  # 创建副本避免修改原数据
                
                # Fetch DNA sequence
                    seq_dna = fetch_dna(fasta_open, chr_name, start, end)
                    
                    # 验证序列长度
                    if len(seq_dna) == 0:
                        print(f"Warning: Empty sequence for index {idx}, skipping")
                        continue
                    
                    # Convert DNA to indices (uint8 encoding, compatible with paddy)
                    seq_indices = dna_to_indices(seq_dna)
                    
                    # Process targets
                    if decimals is not None:
                        target_values = np.around(target_values, decimals=decimals)
                    
                    # Ensure no infinite values and convert to float32
                    target_values = np.nan_to_num(target_values, nan=0.0, posinf=0.0, neginf=0.0)
                    target_values = target_values.astype(np.float32)
                    
                    # 验证数据完整性
                    if seq_indices.size == 0 or target_values.size == 0:
                        print(f"Warning: Invalid data shape for index {idx}, skipping")
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
        
        print(f"Written {tfr_file} with {successful_count}/{len(batch_indices)} sequences")
        return True
        
    except Exception as e:
        print(f"Error processing batch {tfr_file}: {e}")
        return False
    finally:
        # 确保资源被正确释放
        if fasta_open is not None:
            try:
                fasta_open.close()
            except:
                pass


def fetch_dna(fasta_open, chrm, start, end):
    """Fetch DNA when start/end may reach beyond chromosomes."""
    
    # Initialize sequence
    seq_len = end - start
    seq_dna = ""
    
    # Add N's for left over reach
    if start < 0:
        seq_dna = "N" * (-start)
        start = 0
    
    # Get DNA
    seq_dna += fasta_open.fetch(chrm, start, end)
    
    # Add N's for right over reach
    if len(seq_dna) < seq_len:
        seq_dna += "N" * (seq_len - len(seq_dna))
    
    return seq_dna


if __name__ == "__main__":
    main()
    # Example:
    # ./se_data_transfer.py NumChr.Rice_MSUv7.fa Nip_ATGsite_UD16K_Bed.xlsx -p 16
    # ./se_data_transfer.py NumChr.Rice_MSUv7.fa Nip_ATGsite_UD16K_Bed.xlsx -l 32768 -w 32 -c 0 -p 8
    # ./se_data_transfer.py NumChr.Rice_MSUv7.fa Nip_ATGsite_UD16K_Bed.xlsx -t targets.txt -p 16
