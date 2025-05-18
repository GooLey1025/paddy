import glob
import sys
import yaml

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
    ):
        self.data_dir = data_dir
        self.split_label = split_label
        self.batch_size = batch_size
        self.shuffle_buffer = shuffle_buffer
        self.cycle_length = cycle_length
        self.mode = mode
        self.tfr_pattern = tfr_pattern
        self.repeat = repeat

        data_stats_file = f"{self.data_dir}/statistics.yaml"
        with open(data_stats_file) as data_stats_open:
            data_stats = yaml.safe_load(data_stats_open)
        self.num_genes = data_stats["num_genes"]
        self.num_bins = data_stats["num_bins"]
        self.num_tracks = data_stats["num_tracks"]

        self.tfr_path = f"{self.data_dir}/tfrecords/{self.split_label}-*.tfr"

        self.make_dataset(cycle_length=self.cycle_length)

    def generate_parser(self):
        """Generate parser function for TFRecordDataset."""

        def parse_proto(example_protos):
            """Parse TFRecord protobuf."""

            # define features
            features = {
                "features": tf.io.FixedLenFeature([], tf.string),
                "targets": tf.io.FixedLenFeature([], tf.string),
            }

            # parse example into features
            parsed_features = tf.io.parse_single_example(example_protos,
                                                         features=features)

            # decode features (tracks data)
            tracks = tf.io.decode_raw(parsed_features["features"], tf.float32)
            tracks = tf.reshape(tracks, [self.num_tracks, self.num_bins])

            # decode targets (labels)
            targets = tf.io.decode_raw(parsed_features["targets"], tf.float32)
            # Reshape based on your label structure (if known)
            # If using the tissue expression data from the label file
            if hasattr(self, 'tissue_names'):
                targets = tf.reshape(targets, [len(self.tissue_names)])

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

    def distribute(self, strategy):
        """Wrap Dataset to distribute across devices."""
        self.dataset = strategy.experimental_distribute_dataset(self.dataset)
