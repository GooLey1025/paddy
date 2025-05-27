import glob
import sys
import yaml
import numpy as np

from natsort import natsorted
import tensorflow as tf


def file_to_records(filename: str) -> tf.data.Dataset:
    """Read TFRecord file into tf.data.Dataset."""
    return tf.data.TFRecordDataset(filename)


class TracksDataset:
    """Labeled sequence dataset for Tensorflow.

    Args:
      data_dir (str): Dataset directory.
      split_label (str): Dataset split, e.g. train, valid, test.
      batch_size (int): Batch size.
      shuffle_buffer (int): Shuffle buffer size. Defaults to 128.
      mode (str): Dataset mode, e.g. train/eval. Defaults to 'eval'.
      tfr_pattern (str): TFRecord pattern to glob. Defaults to split_label.
      repeat (bool): Whether to repeat the dataset. Defaults to False.
      transpose_input (bool): Whether to transpose input features. Defaults to False.
    """

    def __init__(
        self,
        data_dir: str,
        split_label: str,
        batch_size: int,
        shuffle_buffer: int = 128,
        cycle_length: int = 4,
        mode: str = "eval",
        tfr_pattern: str = None,
        repeat: bool = False,
        transpose_input: bool = False,
    ):
        self.data_dir = data_dir
        self.split_label = split_label
        self.batch_size = batch_size
        self.shuffle_buffer = shuffle_buffer
        self.cycle_length = cycle_length
        self.mode = mode
        self.tfr_pattern = tfr_pattern
        self.repeat = repeat
        self.transpose_input = transpose_input

        data_stats_file = f"{self.data_dir}/statistics.yaml"
        with open(data_stats_file) as data_stats_open:
            data_stats = yaml.safe_load(data_stats_open)
        self.num_genes = data_stats["num_genes"]
        self.num_bins = data_stats["num_bins"]
        self.num_tracks = data_stats["num_tracks"]

        # set default target length and num targets
        self.target_length = data_stats.get("target_length", 1)
        self.num_targets = data_stats["num_targets"]

        self.tfr_path = f"{self.data_dir}/tfrecords/{self.split_label}-*.tfr"

        self.make_dataset(cycle_length=self.cycle_length)

    def generate_parser(self, raw: bool = False):
        """Generate parser function for TFRecordDataset."""

        def parse_proto(example_protos):
            """Parse TFRecord protobuf."""

            # define features
            features = {
                "features": tf.io.FixedLenFeature([], tf.string),
                "targets": tf.io.FixedLenFeature([], tf.string)
            }

            # parse example into features
            parsed_features = tf.io.parse_single_example(example_protos,
                                                         features=features)

            # decode features (tracks data)
            tracks = tf.io.decode_raw(parsed_features["features"], tf.float32)
            if not raw:
                # Reshape to [num_bins, num_tracks] - data is always stored in this format
                tracks = tf.reshape(tracks, [self.num_bins, self.num_tracks])

                # Transpose input if requested
                if self.transpose_input:
                    tracks = tf.transpose(
                        tracks)  # Reshape to [num_tracks, num_bins]

            # decode targets (labels)
            targets = tf.io.decode_raw(parsed_features["targets"], tf.float32)
            # Reshape based on your label structure (if known)
            # If using the tissue expression data from the label file

            # if hasattr(self, 'tissue_names'):
            #     targets = tf.reshape(targets, [len(self.tissue_names)])

            return tracks, targets

        return parse_proto

    def make_dataset(self, cycle_length):
        """Make dataset from TFRecord files."""

        # initialize dataset from TFRecord files
        tfr_files = natsorted(glob.glob(self.tfr_path))
        if tfr_files:
            dataset = tf.data.Dataset.from_tensor_slices(tfr_files)
        else:
            print("Cannot order TFRecords %s" % self.tfr_path, file=sys.stderr)
            dataset = tf.data.Dataset.list_files(self.tfr_path)

        # train
        if self.mode == "train":
            # repeat if requested or in train mode
            if self.repeat:
                dataset = dataset.repeat()

            # interleave files
            dataset = dataset.interleave(
                map_func=file_to_records,
                cycle_length=cycle_length,
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )

            dataset = dataset.shuffle(buffer_size=self.shuffle_buffer,
                                      reshuffle_each_iteration=True)

        # valid/test
        else:
            dataset = dataset.flat_map(file_to_records)
            # repeat validation/test datasets if requested
            if self.repeat:
                dataset = dataset.repeat()

        # map parser
        dataset = dataset.map(self.generate_parser(),
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # batch
        dataset = dataset.batch(self.batch_size)

        # prefetch
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        # hold on
        self.dataset = dataset

    def batches_per_epoch(self):
        """Compute number of batches per epoch."""
        return self.num_genes // self.batch_size

    def numpy(
        self,
        return_inputs=True,
        return_outputs=True,
        step=1,
        target_slice=None,
        dtype="float16",
    ):
        """Convert TFR inputs and/or outputs to numpy arrays."""
        with tf.name_scope("numpy"):
            # initialize dataset from TFRecords glob
            tfr_files = natsorted(glob.glob(self.tfr_path))
            if tfr_files:
                # dataset = tf.data.Dataset.list_files(tf.constant(tfr_files), shuffle=False)
                dataset = tf.data.Dataset.from_tensor_slices(tfr_files)
            else:
                print("Cannot order TFRecords %s" % self.tfr_path,
                      file=sys.stderr)
                dataset = tf.data.Dataset.list_files(self.tfr_path)

            # read TF Records
            dataset = dataset.flat_map(file_to_records)
            dataset = dataset.map(self.generate_parser(raw=True))
            dataset = dataset.batch(1)

        # initialize inputs and outputs
        tracks_input = []
        targets = []

        # collect inputs and outputs
        for tracks_raw, targets_raw in dataset:
            # input tracks
            if return_inputs:
                # For chromatin accessibility data
                if not self.transpose_input:
                    # shape [1, num_bins, num_tracks]
                    tracks = tracks_raw.numpy()
                    tracks = tracks.reshape((self.num_bins, self.num_tracks))
                else:
                    # shape [1, num_tracks, num_bins]
                    tracks = tracks_raw.numpy()
                    tracks = tracks.reshape((self.num_tracks, self.num_bins))

                # Apply cropping if needed
                if hasattr(self, 'crop_len') and self.crop_len > 0:
                    crop_len = self.crop_len // 2
                    if not self.transpose_input:
                        tracks = tracks[crop_len:-crop_len, :]
                    else:
                        tracks = tracks[:, crop_len:-crop_len]

                tracks_input.append(tracks)

            # targets
            if return_outputs:
                targets1 = targets_raw.numpy().astype(dtype)
                # Reshape based on expected target dimensions
                targets1 = np.reshape(targets1, (self.target_length, -1))
                if target_slice is not None:
                    targets1 = targets1[:, target_slice]
                if step > 1:
                    step_i = np.arange(0, self.target_length, step)
                    targets1 = targets1[step_i, :]
                targets.append(targets1)

        # make arrays
        if return_inputs:
            tracks_input = np.array(tracks_input)
        if return_outputs:
            targets = np.array(targets, dtype=dtype)

        # return
        if return_inputs and return_outputs:
            return tracks_input, targets
        elif return_inputs:
            return tracks_input
        else:
            return targets

    def distribute(self, strategy):
        """Wrap Dataset to distribute across devices."""
        self.dataset = strategy.experimental_distribute_dataset(self.dataset)
