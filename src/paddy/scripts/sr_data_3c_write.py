#!/usr/bin/env python3
"""
Single batch TFRecord writer for SloRice 3-class classification.

This script processes a single batch of SNP data and writes it to a TFRecord file.
It accepts base64 encoded gzipped batch data as a command line argument.

The script handles strand information and includes it in the TFRecord features.
"""

import argparse
import os
import sys
import pickle
from typing import Dict, List, Any

import numpy as np
import tensorflow as tf


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Write a single TFRecord batch for SloRice 3-class classification',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('batch_data', type=str, help='Base64 encoded gzipped batch data, or a path to a file containing it')
    parser.add_argument('output_tfr', type=str, help='Output TFRecord file path')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    return parser.parse_args()


def dna_to_indices(seq_dna: str) -> np.ndarray:
    """Convert DNA sequence to indices for uint8 encoding."""
    dna_alphabet = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
    seq_dna = seq_dna.upper()
    seq_indices = np.zeros(len(seq_dna), dtype=np.uint8)
    seq_indices.fill(4)  # Default to N (4)
    
    for base, idx in dna_alphabet.items():
        seq_indices[np.array([c == base for c in seq_dna])] = idx
    
    return seq_indices


def feature_bytes(values: np.ndarray) -> tf.train.Feature:
    """Convert numpy arrays to bytes features."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values.tobytes()]))


def feature_int64(value: int) -> tf.train.Feature:
    """Convert an integer value to int64 feature."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def write_batch_tfrecord(batch_data: List[Dict[str, Any]], output_file: str, verbose: bool = False) -> int:
    """Write a batch of SNP data to TFRecord file.
    
    Args:
        batch_data: List of dictionaries containing 'ref_seq', 'alt_seq', 'class_label', 'strand'
        output_file: Path to output TFRecord file
        verbose: Whether to print verbose output
        
    Returns:
        Number of successfully processed sequences
    """
    tf_opts = tf.io.TFRecordOptions(compression_type="ZLIB")
    successful_count = 0
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with tf.io.TFRecordWriter(output_file, tf_opts) as writer:
        for i, snp_data in enumerate(batch_data):
            try:
                ref_seq = snp_data['ref_seq']
                alt_seq = snp_data['alt_seq']
                class_label = snp_data['class_label']
                strand = snp_data['strand']
                
                # Convert DNA to indices
                ref_indices = dna_to_indices(ref_seq)
                alt_indices = dna_to_indices(alt_seq)
                class_label_int = np.int64(class_label)
                
                # Convert strand to integer (0 for '+', 1 for '-')
                strand_int = np.int64(0 if strand == '+' else 1)
                
                # Create features
                features_dict = {
                    "ref_sequence": feature_bytes(ref_indices),
                    "alt_sequence": feature_bytes(alt_indices),
                    "class_label": feature_int64(class_label_int),
                    "strand": feature_int64(strand_int),
                }
                
                example = tf.train.Example(features=tf.train.Features(feature=features_dict))
                writer.write(example.SerializeToString())
                successful_count += 1
                
                if verbose and (i + 1) % 100 == 0:
                    print(f"Processed {i + 1}/{len(batch_data)} sequences")
                    
            except Exception as e:
                raise ValueError(f"Error processing sequence {i}: {e}")
    
    return successful_count


def main() -> None:
    args = parse_args()
    
    if args.verbose:
        print(f"Decoding batch data...")
    
    # Decode batch data from base64 encoded gzipped string
    try:
        import base64
        import gzip
        
        # Support passing a file path instead of inline data
        if os.path.isfile(args.batch_data):
            with open(args.batch_data, 'r') as f:
                batch_b64_str = f.read()
        else:
            batch_b64_str = args.batch_data

        # Decode from base64
        compressed = base64.b64decode(batch_b64_str.encode('ascii'))
        # Decompress
        batch_bytes = gzip.decompress(compressed)
        # Deserialize
        batch_data = pickle.loads(batch_bytes)
    except Exception as e:
        print(f"Error decoding batch data: {e}", file=sys.stderr)
        sys.exit(1)
    
    if args.verbose:
        print(f"Loaded {len(batch_data)} sequences")
        print(f"Writing to {args.output_tfr}")
    
    # Write TFRecord
    successful_count = write_batch_tfrecord(batch_data, args.output_tfr, args.verbose)
    
    if args.verbose:
        print(f"Successfully wrote {successful_count}/{len(batch_data)} sequences to {args.output_tfr}")
    
    if successful_count == 0:
        print(f"Error: No sequences were successfully written to {args.output_tfr}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
