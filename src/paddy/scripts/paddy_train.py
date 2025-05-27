#!/usr/bin/env python3

import argparse
import os
import shutil
import yaml
import tensorflow as tf
from paddy import seqnn
from paddy import dataset
from paddy import trainer
from paddy import dataset

from tensorflow.keras import mixed_precision


def main():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("params_file", help="YAML file with model parameters")
    parser.add_argument(
        "-k",
        "--keras_fit",
        action="store_true",
        default=False,
        help="Train with Keras fit method [Default: %(default)s]",
    )
    parser.add_argument(
        "-m",
        "--mixed_precision",
        action="store_true",
        default=False,
        help="Train with mixed precision [Default: %(default)s]",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        default="train_out",
        help="Output directory [Default: %(default)s]",
    )
    parser.add_argument(
        "--tfr_train",
        default=None,
        help=
        "Training TFR pattern string appended to data_dir/tfrecords [Default: %(default)s]",
    )
    parser.add_argument(
        "--tfr_eval",
        default=None,
        help=
        "Evaluation TFR pattern string appended to data_dir/tfrecords [Default: %(default)s]",
    )
    parser.add_argument(
        "-l",
        "--log_dir",
        default="log_out",
        help="Tensorboard log directory [Default: %(default)s]",
    )
    parser.add_argument(
        "--transpose_input",
        default=False,
        action="store_true",
        help=
        "Transpose input features from [num_bins, num_tracks] to [num_tracks, num_bins] [Default: %(default)s]",
    )
    parser.add_argument(
        "--restore",
        action="store_true",
        default=False,
        help="Restore model from checkpoint in out_dir [Default: %(default)s]",
    )
    parser.add_argument(
        "--restore_step",
        type=int,
        default=None,
        help=
        "Specific checkpoint step to restore (will use latest if not specified) [Default: %(default)s]",
    )
    parser.add_argument("data_dirs",
                        nargs="+",
                        help="Train/valid/test data directorie(s)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.params_file != "%s/params.yaml" % args.out_dir:
        shutil.copy(args.params_file, "%s/params.yaml" % args.out_dir)

    with open(args.params_file, "r") as f:
        params = yaml.safe_load(f)
    params_model = params["model"]
    params_train = params["train"]

    train_data = []
    eval_data = []
    strand_pairs = []

    for data_dir in args.data_dirs:
        train_data.append(
            dataset.TracksDataset(data_dir,
                                  split_label="train",
                                  batch_size=params_train["batch_size"],
                                  shuffle_buffer=params_train.get(
                                      "shuffle_buffer", 128),
                                  mode="train",
                                  tfr_pattern=args.tfr_train,
                                  repeat=True,
                                  transpose_input=args.transpose_input))

        eval_data.append(
            dataset.TracksDataset(
                data_dir,
                split_label="valid",
                batch_size=params_train["batch_size"],
                mode="eval",
                tfr_pattern=args.tfr_eval,
                transpose_input=args.transpose_input,
            ))

    if args.mixed_precision:
        mixed_precision.set_global_policy("mixed_bfloat16")

    if params_train.get("num_gpu", 1) == 1:
        ########################################
        # one GPU

        # initialize model
        params_model["transpose_input"] = args.transpose_input
        seqnn_model = seqnn.SeqNN(params_model)

        seqnn_trainer = trainer.Trainer(
            params_train,
            train_data,
            eval_data,
            args.out_dir,
            args.log_dir,
        )
        seqnn_trainer.compile(seqnn_model)

        checkpoint_dir = args.out_dir

        if args.restore and os.path.exists(checkpoint_dir):
            if args.restore_step is not None:
                checkpoint_path = os.path.join(checkpoint_dir,
                                               f"ckpt-{args.restore_step}")
                if os.path.exists(checkpoint_path + ".index"):
                    print(
                        f"Restoring model from checkpoint step {args.restore_step}..."
                    )
                    seqnn_trainer.restore(checkpoint_path, seqnn_model)
                else:
                    print(
                        f"Specified checkpoint {checkpoint_path} not found, starting from scratch."
                    )
            else:
                latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
                if latest_checkpoint is not None:
                    print(
                        f"Restoring model from latest checkpoint {latest_checkpoint}..."
                    )
                    seqnn_trainer.restore(latest_checkpoint, seqnn_model)
                else:
                    print(
                        f"No checkpoints found in {checkpoint_dir}, starting from scratch."
                    )

    else:
        ########################################
        # multi GPU
        print("Multi GPU training not implemented yet.")
        exit()

    # train model

    if len(args.data_dirs) == 1:
        seqnn_trainer.fit_tape(seqnn_model)
    else:
        print("fit2 not implemented yet.")
        exit()
        seqnn_trainer.fit2(seqnn_model)


if __name__ == "__main__":
    main()
