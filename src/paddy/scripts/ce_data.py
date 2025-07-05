#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import os
import glob
import h5py
import pandas as pd
from tqdm import tqdm
from natsort import natsorted
import yaml
import argparse


class DataGenerator:

    def __init__(self,
                 h5_dir,
                 output_dir,
                 xlsx_file=None,
                 train_ratio=0.8,
                 valid_ratio=0.1,
                 test_ratio=0.1,
                 test_chrom=None,
                 valid_chrom=None):
        """
        Initialize the data generator
        Args:
            h5_dir: h5 file directory
            output_dir: output TFRecord file directory
            xlsx_file: path to Excel file containing gene metadata and expression data
            train_ratio: ratio of training data
            valid_ratio: ratio of validation data
            test_ratio: ratio of test data
            test_chrom: chromosome to use for test set (overrides test_ratio)
            valid_chrom: chromosome to use for validation set (overrides valid_ratio)
        """
        self.h5_dir = h5_dir
        self.output_dir = output_dir
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio
        self.xlsx_file = xlsx_file
        self.test_chrom = test_chrom
        self.valid_chrom = valid_chrom
        self.labels = None
        self.tissue_names = None
        self.gene_metadata = None  # To store the first 7 columns of xlsx file
        self.using_chromosome_split = (test_chrom is not None
                                       or valid_chrom is not None)

        # Initialize count variables
        self.train_count = 0
        self.valid_count = 0
        self.test_count = 0

        # Ensure ratios sum to 1
        total_ratio = train_ratio + valid_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-10:
            print(f"Warning: ratios sum to {total_ratio}, normalizing to 1.0")
            self.train_ratio /= total_ratio
            self.valid_ratio /= total_ratio
            self.test_ratio /= total_ratio

        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Automatically detect num_bins and num_tracks
        self._detect_dimensions()

        # Load labels if provided
        if xlsx_file:
            self._load_xlsx_data()

    def _load_xlsx_data(self):
        """
        Load data from Excel file, using rows from 8 onwards as labels
        and preserving first 7 columns as gene metadata
        """
        # Load the Excel file
        print(f"Loading data from {self.xlsx_file}")
        excel_data = pd.read_excel(self.xlsx_file, header=0)

        # Extract gene metadata (first 7 columns)
        self.gene_metadata = excel_data.iloc[:, :7].copy()
        print(
            f"Extracted metadata for {len(self.gene_metadata)} genes (first 7 columns)"
        )

        # Extract expression data (columns from 8 onwards)
        expression_data = excel_data.iloc[:, 7:]

        # Store tissue names (column headers)
        self.tissue_names = expression_data.columns.tolist()
        print(
            f"Loaded {len(self.tissue_names)} tissue/cell types from {self.xlsx_file}"
        )

        # Convert to numpy array for easier access
        self.labels = expression_data.values
        print(f"Loaded {self.labels.shape[0]} gene expression profiles")

        # Check if number of rows in labels matches expected gene count
        expected_genes = self._count_total_genes()
        if len(self.labels) != expected_genes:
            raise ValueError(
                f"Label count ({len(self.labels)}) doesn't match expected gene count ({expected_genes})"
            )

    def _count_total_genes(self):
        """
        Count the total number of genes across all h5 files
        """
        h5_files = natsorted(glob.glob(os.path.join(self.h5_dir, "*.h5")))

        # Use the first file to determine genes per file
        with h5py.File(h5_files[0], 'r') as f:
            genes_per_file = f['targets'].shape[0]

        # All files should have the same number of genes
        total_genes = genes_per_file

        return total_genes

    def _detect_dimensions(self):
        """
        Automatically detect the dimensions (num_bins and num_tracks) from h5 files
        """
        # Get the first h5 file
        h5_files = natsorted(glob.glob(os.path.join(self.h5_dir, "*.h5")))
        if not h5_files:
            raise ValueError(f"No h5 files found in {self.h5_dir}")

        # Read the first file
        first_file = h5_files[0]
        try:
            data = self.read_h5_file(first_file)
            if data is None:
                raise ValueError(f"Could not read first h5 file: {first_file}")

            # Get dimension information
            self.num_genes = data.shape[0]
            self.num_bins = data.shape[1]

            # Calculate the number of tracks
            self.num_tracks = len(h5_files)

            print(f"Detected dimensions: num_genes={self.num_genes}, "
                  f"num_bins={self.num_bins}, num_tracks={self.num_tracks}")

        except Exception as e:
            raise ValueError(
                f"Error detecting dimensions from {first_file}: {e}")

    def read_h5_file(self, file_path):
        """
        Read a single h5 file
        Args:
            file_path: Path to the h5 file
        Returns:
            numpy array of shape (num_genes, num_bins)
        """
        # Using h5py to read HDF5 files

        with h5py.File(file_path, 'r') as h5_file:
            # Directly access the 'targets' dataset based on file inspection
            if 'targets' in h5_file:
                data = h5_file['targets'][:]
                return data
            # Fallback: try to get the first dataset if 'targets' doesn't exist
            else:
                for key in h5_file.keys():
                    if isinstance(h5_file[key], h5py.Dataset):
                        print(
                            f"Warning: 'targets' not found, using '{key}' instead in {file_path}"
                        )
                        data = h5_file[key][:]
                        return data

                raise ValueError(f"No dataset found in {file_path}")

    def _bytes_feature(self, value):
        """Convert numpy array to bytes feature"""
        return tf.train.Feature(bytes_list=tf.train.BytesList(
            value=[value.tobytes()]))

    def _float_feature(self, value):
        """Convert float/double values to float feature"""
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def create_tfrecord_example(self, features_data, targets_data):
        """
        Create a TFRecord example for a single gene
        Args:
            features_data: numpy array of shape (num_tracks, num_bins) - features
            targets_data: numpy array of targets/labels
        Returns:
            tf.train.Example object
        """
        # Always transpose features from (num_tracks, num_bins) to (num_bins, num_tracks)
        features_data = features_data.transpose(
        )  # Now shape: (num_bins, num_tracks)

        features_dict = {
            "features": self._bytes_feature(features_data.astype(np.float32)),
            "targets": self._bytes_feature(targets_data.astype(np.float32))
        }

        return tf.train.Example(features=tf.train.Features(
            feature=features_dict))

    def create_targets_file(self, indices, split_mappings):
        """
        Create targets.txt file with gene metadata and dataset split information
        Args:
            indices: Original indices of all genes in the dataset
            split_mappings: Dictionary mapping indices to their dataset split (train/valid/test)
        """
        if self.gene_metadata is None:
            print(
                "Warning: No gene metadata available, skipping targets.txt creation"
            )
            return

        # Create a DataFrame for targets.txt
        targets_df = self.gene_metadata.copy()

        # Add a column to indicate dataset split
        targets_df['split'] = pd.Series([split_mappings[i] for i in indices],
                                        index=targets_df.index)

        # Save to file
        targets_file_path = os.path.join(self.output_dir, "targets.txt")
        targets_df.to_csv(targets_file_path, sep='\t', index=True)
        print(f"Created targets.txt file at {targets_file_path}")

        return targets_df

    def split_by_chromosome(self):
        """
        Split indices by chromosome
        Returns:
            train_indices, valid_indices, test_indices: Lists of indices for each split
        """
        if self.gene_metadata is None or 'Chrom' not in self.gene_metadata.columns:
            print(
                "Warning: Cannot split by chromosome - no chromosome information found"
            )
            return None, None, None

        # Get chromosome column
        chrom_col = self.gene_metadata['Chrom']
        original_indices = np.arange(self.num_genes)

        # Find indices for each specified chromosome
        test_indices = []
        valid_indices = []
        train_indices = []

        for i in original_indices:
            chrom = chrom_col.iloc[i]

            if self.test_chrom is not None and str(chrom) == str(
                    self.test_chrom):
                test_indices.append(i)
            elif self.valid_chrom is not None and str(chrom) == str(
                    self.valid_chrom):
                valid_indices.append(i)
            else:
                train_indices.append(i)

        # Shuffle each set of indices
        np.random.shuffle(test_indices)
        np.random.shuffle(valid_indices)
        np.random.shuffle(train_indices)

        # Print statistics
        print(f"Split by chromosome:")
        print(f"  Train: {len(train_indices)} genes")
        print(
            f"  Valid: {len(valid_indices)} genes (chromosome {self.valid_chrom})"
        )
        print(
            f"  Test: {len(test_indices)} genes (chromosome {self.test_chrom})"
        )

        # Update instance variable to indicate we're using chromosome split
        self.using_chromosome_split = True
        self.train_count = len(train_indices)
        self.valid_count = len(valid_indices)
        self.test_count = len(test_indices)

        return train_indices, valid_indices, test_indices

    def process_data(self, batch_size=64):
        """
        Process all h5 files and generate TFRecord files with train/validation/test splits
        Args:
            batch_size: number of genes in each TFRecord file
        """

        h5_files = natsorted(glob.glob(os.path.join(self.h5_dir, "*.h5")))
        if not h5_files:
            raise ValueError(f"No h5 files found in {self.h5_dir}")

        print(
            f"Found {len(h5_files)} h5 files. Expected to be {self.num_tracks} tracks."
        )

        first_h5 = h5_files[0]

        # Create a 3D array to hold all data: (num_genes, num_tracks, num_bins)
        all_data = np.zeros((self.num_genes, self.num_tracks, self.num_bins),
                            dtype=np.float32)

        # Fill the 3D array with data from each h5 file
        for track_idx, h5_file in enumerate(
                tqdm(h5_files, desc="Reading h5 files")):
            track_data = self.read_h5_file(h5_file)
            if track_data is not None:
                all_data[:, track_idx, :] = track_data

        print(f"Data matrix shape: {all_data.shape}")
        print(
            f"Data will be stored with shape (num_bins, num_tracks): features will be transposed"
        )

        # Get original indices for all genes
        original_indices = np.arange(self.num_genes)

        # Split indices by method (chromosome or random)
        if self.test_chrom is not None or self.valid_chrom is not None:
            # Split by chromosome
            train_indices, valid_indices, test_indices = self.split_by_chromosome(
            )
            if train_indices is None:
                # Fallback to random split if chromosome split failed
                print("Falling back to random split")
                use_random_split = True
            else:
                use_random_split = False
        else:
            # Use random split
            use_random_split = True

        if use_random_split:
            # Shuffle indices for random split
            shuffled_indices = np.arange(self.num_genes)
            np.random.shuffle(shuffled_indices)

            # Calculate split points
            train_end = int(self.num_genes * self.train_ratio)
            valid_end = train_end + int(self.num_genes * self.valid_ratio)

            # Split indices
            train_indices = shuffled_indices[:train_end]
            valid_indices = shuffled_indices[train_end:valid_end]
            test_indices = shuffled_indices[valid_end:]

            print(f"Random split:")
            print(
                f"  Train: {len(train_indices)} genes ({self.train_ratio:.1%})"
            )
            print(
                f"  Valid: {len(valid_indices)} genes ({self.valid_ratio:.1%})"
            )
            print(f"  Test: {len(test_indices)} genes ({self.test_ratio:.1%})")

            # Store counts for statistics
            self.train_count = len(train_indices)
            self.valid_count = len(valid_indices)
            self.test_count = len(test_indices)

        # Create mapping of original indices to their split assignment
        split_mappings = {}
        for idx in train_indices:
            split_mappings[idx] = "train"
        for idx in valid_indices:
            split_mappings[idx] = "valid"
        for idx in test_indices:
            split_mappings[idx] = "test"

        # Create targets.txt file with gene metadata and split information
        self.create_targets_file(original_indices, split_mappings)

        # Create statistics file
        self._create_statistics()

        # Process each split
        print(
            f"Creating train TFRecord files ({len(train_indices)} examples)..."
        )
        self._write_tfrecords(all_data, train_indices, "train", batch_size)

        print(
            f"Creating validation TFRecord files ({len(valid_indices)} examples)..."
        )
        self._write_tfrecords(all_data, valid_indices, "valid", batch_size)

        print(
            f"Creating test TFRecord files ({len(test_indices)} examples)...")
        self._write_tfrecords(all_data, test_indices, "test", batch_size)

    def _write_tfrecords(self, all_data, indices, split_name, batch_size):
        """
        Write TFRecord files for a specific data split
        Args:
            all_data: complete dataset array of shape (num_genes, num_tracks, num_bins)
            indices: indices to use for this split
            split_name: name of the split (train, valid, test)
            batch_size: number of examples per TFRecord file
        """
        os.makedirs(os.path.join(self.output_dir, "tfrecords"), exist_ok=True)
        num_examples = len(indices)
        num_tfrecords = (num_examples + batch_size - 1) // batch_size

        for i in tqdm(range(num_tfrecords),
                      desc=f"Writing {split_name} TFRecord files"):
            start_batch = i * batch_size
            end_batch = min((i + 1) * batch_size, num_examples)
            batch_indices = indices[start_batch:end_batch]

            output_path = os.path.join(self.output_dir,
                                       f'tfrecords/{split_name}-{i}.tfr')

            with tf.io.TFRecordWriter(output_path) as writer:
                for idx in batch_indices:
                    # Get features for this gene (all tracks)
                    features_data = all_data[
                        idx]  # Shape: (num_tracks, num_bins)

                    # Get targets for this gene if available
                    targets_data = None
                    if self.labels is not None and idx < len(self.labels):
                        targets_data = self.labels[idx]

                    example = self.create_tfrecord_example(
                        features_data, targets_data)
                    writer.write(example.SerializeToString())

    def _create_statistics(self):
        """
        Create statistics file
        """
        statistics = {
            "num_genes": self.num_genes,
            "num_bins": self.num_bins,
            "num_tracks": self.num_tracks,
            "data_format": "transposed: [num_bins, num_tracks]"
        }

        # Add split method and related statistics
        if hasattr(self,
                   'using_chromosome_split') and self.using_chromosome_split:
            statistics["split_method"] = "chromosome"
            statistics["test_chromosome"] = self.test_chrom
            statistics["valid_chromosome"] = self.valid_chrom
            statistics["train_seqs"] = self.train_count
            statistics["valid_seqs"] = self.valid_count
            statistics["test_seqs"] = self.test_count
        else:
            statistics["split_method"] = "random"
            statistics["train_ratio"] = self.train_ratio
            statistics["valid_ratio"] = self.valid_ratio
            statistics["test_ratio"] = self.test_ratio
            statistics["train_seqs"] = self.train_count
            statistics["valid_seqs"] = self.valid_count
            statistics["test_seqs"] = self.test_count

        # Add information about the number of target tissues if available
        if self.tissue_names is not None:
            statistics["num_targets"] = len(self.tissue_names)
            statistics["target_names"] = self.tissue_names

        # Save as YAML file
        with open(os.path.join(self.output_dir, "statistics.yaml"), "w") as f:
            yaml.dump(statistics, f, default_flow_style=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--h5_dir",
        type=str,
        default="/home/gl/projects/Borzoi/paddy/data/seqs_cov",
        help=("Each h5 file stores one chromation track info of the genes,"
              " e.g if you have five h5 files in h5_dir, means five tracks."
              " Of course, they share same genes and same order."))
    parser.add_argument(
        "--output_dir",
        type=str,
        default="test",
        help=("Output directory,"
              "also used as data directory for training paddy_train.py"))
    parser.add_argument(
        "--xlsx_file",
        type=str,
        default=None,
        help=
        "Excel file containing gene metadata (first 7 columns) and expression data (8th column onwards)"
    )
    parser.add_argument("--train_ratio",
                        type=float,
                        default=0.8,
                        help="Training set ratio")
    parser.add_argument("--valid_ratio",
                        type=float,
                        default=0.1,
                        help="Validation set ratio")
    parser.add_argument("--test_ratio",
                        type=float,
                        default=0.1,
                        help="Test set ratio")
    parser.add_argument(
        "--valid_chrom",
        type=str,
        default=None,
        help="Chromosome to use for validation set (overrides valid_ratio)")
    parser.add_argument(
        "--test_chrom",
        type=str,
        default=None,
        help="Chromosome to use for test set (overrides test_ratio)")

    args = parser.parse_args()

    # Create data generator with split ratios and xlsx file
    data_generator = DataGenerator(args.h5_dir,
                                   args.output_dir,
                                   xlsx_file=args.xlsx_file,
                                   train_ratio=args.train_ratio,
                                   valid_ratio=args.valid_ratio,
                                   test_ratio=args.test_ratio,
                                   test_chrom=args.test_chrom,
                                   valid_chrom=args.valid_chrom)

    # Process data
    data_generator.process_data()


if __name__ == "__main__":
    main()
