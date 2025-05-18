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
                 label_file=None,
                 train_ratio=0.8,
                 valid_ratio=0.1,
                 test_ratio=0.1):
        """
        Initialize the data generator
        Args:
            h5_dir: h5 file directory
            output_dir: output TFRecord file directory
            label_file: path to label file (CSV format)
            train_ratio: ratio of training data
            valid_ratio: ratio of validation data
            test_ratio: ratio of test data
        """
        self.h5_dir = h5_dir
        self.output_dir = output_dir
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio
        self.label_file = label_file
        self.labels = None
        self.tissue_names = None

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
        if label_file:
            self._load_labels()
        
        self._create_statistics()

    def _load_labels(self):
        """
        Load labels from CSV file
        """
        # Load the labels file directly as CSV
        labels_df = pd.read_csv(self.label_file, sep='\t')

        # Store tissue names (column headers)
        self.tissue_names = labels_df.columns.tolist()
        print(
            f"Loaded {len(self.tissue_names)} tissue/cell types from {self.label_file}"
        )

        # Convert to numpy array for easier access
        self.labels = labels_df.values
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

            print(
                f"Detected dimensions: num_genes={self.num_genes}, "
                f"num_bins={self.num_bins}, num_tracks={self.num_tracks}"
            )

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
        features_dict = {
            "features": self._bytes_feature(features_data.astype(np.float32)),
            "targets": self._bytes_feature(targets_data.astype(np.float32))
        }

        return tf.train.Example(features=tf.train.Features(feature=features_dict))

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

        # Read the gene count from the first file
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

        # Shuffle the indices to randomize the data
        indices = np.arange(self.num_genes)
        np.random.shuffle(indices)

        # Calculate the split points
        train_end = int(self.num_genes * self.train_ratio)
        valid_end = train_end + int(self.num_genes * self.valid_ratio)

        # Split indices into train/valid/test sets
        train_indices = indices[:train_end]
        valid_indices = indices[train_end:valid_end]
        test_indices = indices[valid_end:]

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
            "train_ratio": self.train_ratio,
            "valid_ratio": self.valid_ratio,
            "test_ratio": self.test_ratio
        }
        with open(os.path.join(self.output_dir, "statistics.yaml"), "w") as f:
            yaml.dump(statistics, f, default_flow_style=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--h5_dir", type=str, 
        default="/home/gl/projects/Borzoi/paddy/data/seqs_cov"
        )
    parser.add_argument(
        "--output_dir", type=str, 
        default="test",
        help=("Output directory,"
              "also used as data directory for training paddy_train.py"
        )
    )
    parser.add_argument(
        "--label_file", type=str, 
        default="2_P8_Nip8_23tissues.Exp"
        )
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Training set ratio")
    parser.add_argument("--valid_ratio", type=float, default=0.1,
                        help="Validation set ratio")
    parser.add_argument("--test_ratio", type=float, default=0.1,
                        help="Test set ratio")
    args = parser.parse_args()

    # Create data generator with split ratios and label file
    data_generator = DataGenerator(args.h5_dir,
                                   args.output_dir,
                                   label_file=args.label_file,
                                   train_ratio=args.train_ratio,
                                   valid_ratio=args.valid_ratio,
                                   test_ratio=args.test_ratio
                                   )

    # Process data
    data_generator.process_data()

if __name__ == "__main__":
    main()
