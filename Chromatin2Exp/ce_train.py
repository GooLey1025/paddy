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
        "-m",
        "--mixed_precision",
        action="store_true",
        default=False,
        help="Train with mixed precision if your gpu supports it. \n"
        " If not supported, float16 will still be used to reduce gpu memory usage. \n"
        " In theory, it is recommended to enable it accompanied by --loss_scale. \n"
        " [Default: %(default)s]",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        default="train_out",
        help="Output directory [Default: %(default)s]",
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
        help=
        "Restore model from checkpoint in args.out_dir (if not specified, will start training from scratch) [Default: %(default)s]",
    )
    parser.add_argument(
        "--loss_scale",
        action="store_true",
        default=False,
        help="Use loss scale for training. A little bit slower but more stable. [Default: %(default)s]",
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
    params_model["transpose_input"] = args.transpose_input

    train_data = []
    eval_data = []
    strand_pairs = []

    for data_dir in args.data_dirs:
        train_data.append(
            dataset.TracksDataset(
                data_dir,
                split_label="train",
                batch_size=params_train["batch_size"],
                shuffle_buffer=params_train.get("shuffle_buffer", 128),
                mode="train",
                repeat=True,
                transpose_input=args.transpose_input,
            )
        )

        eval_data.append(
            dataset.TracksDataset(
                data_dir,
                split_label="valid",
                batch_size=params_train["batch_size"],
                mode="eval",
                transpose_input=args.transpose_input,
            ))

    if args.mixed_precision:
        mixed_precision.set_global_policy("mixed_bfloat16")

    if params_train.get("num_gpu", 1) == 1:
        ########################################
        # one GPU

        # initialize model
        seqnn_model = seqnn.TracksNN(params_model)

        seqnn_trainer = trainer.Trainer(
            params_train,
            train_data,
            eval_data,
            args.out_dir,
            args.log_dir,
            loss_scale=args.loss_scale
        )
        seqnn_trainer.compile(seqnn_model)

        checkpoint_dir = args.out_dir

    else:
        ########################################
        # multi GPU
        print("Multi GPU training not implemented yet.")
        exit()

    # train model

    if len(args.data_dirs) == 1:
        seqnn_trainer.fit_tape(seqnn_model, restore=args.restore)
    else:
        print("fit2 not implemented yet.")
        exit()
        seqnn_trainer.fit2(seqnn_model)


if __name__ == "__main__":
    main()
