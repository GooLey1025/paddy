#!/usr/bin/env python

import argparse
import json
import os
from paddy.utils import set_gpu_device
set_gpu_device()

import shutil
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import mixed_precision

from paddy import dataset
from paddy import seqnn
from paddy import trainer

"""
se_train.py

Train seq2exp model using given parameters and data.
"""


def main():
    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument(
        "-g",
        "--gpu",
        "--visible_device",
        type=int,
        default=0,
        help="comma-separated list of GPU IDs",
    )
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
        "-l",
        "--log_dir",
        default="log_out",
        help="Tensorboard log directory [Default: %(default)s]",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume training from checkpoint [Default: %(default)s]",
    )
    parser.add_argument(
        "--restore",
        default=None,
        help="Restore model and continue training [Default: %(default)s]",
    )
    parser.add_argument(
        "--trunk",
        action="store_true",
        default=False,
        help="Restore only model trunk [Default: %(default)s]",
    )
    parser.add_argument(
        "--tfr_train",
        default=None,
        help="Training TFR pattern string appended to data_dir/tfrecords [Default: %(default)s]",
    )
    parser.add_argument(
        "--tfr_eval",
        default=None,
        help="Evaluation TFR pattern string appended to data_dir/tfrecords [Default: %(default)s]",
    )

    parser.add_argument("params_file", help="YAML file with model parameters")
    parser.add_argument(
        "data_dirs", nargs="+", help="Train/valid/test data directorie(s)"
    )
    args = parser.parse_args()

    if args.keras_fit and len(args.data_dirs) > 1:
        print("Cannot use keras fit method with multi-genome training.")
        exit()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.params_file != "%s/params.yaml" % args.out_dir:
        shutil.copy(args.params_file, "%s/params.yaml" % args.out_dir)

    # read model parameters
    with open(args.params_file) as params_open:
        params = yaml.safe_load(params_open)
    params_model = params["model"]
    params_train = params["train"]

    # read datasets
    train_data = []
    eval_data = []
    strand_pairs = []

    for data_dir in args.data_dirs:
        # set strand pairs
        targets_df = pd.read_csv("%s/targets.txt" % data_dir, sep="\t", index_col=0)
        if "strand_pair" in targets_df.columns:
            strand_pairs.append(np.array(targets_df.strand_pair))

        # load train data
        train_data.append(
            dataset.SeqDataset(
                data_dir,
                split_label="train",
                batch_size=params_train["batch_size"],
                shuffle_buffer=params_train.get("shuffle_buffer", 128),
                mode="train",
                tfr_pattern=args.tfr_train,
                model_type=params_model["model_type"],
            )
        )

        # load eval data
        eval_data.append(
            dataset.SeqDataset(
                data_dir,
                split_label="valid",
                batch_size=params_train["batch_size"],
                mode="eval",
                tfr_pattern=args.tfr_eval,
                model_type=params_model["model_type"],
            )
        )

    params_model["strand_pair"] = strand_pairs

    if args.mixed_precision:
        mixed_precision.set_global_policy("mixed_float16")

    if params_train.get("num_gpu", 1) == 1:
        ########################################
        # one GPU

        # initialize model
        seqnn_model = seqnn.SeqNN(params_model)

        # restore
        if args.restore:
            seqnn_model.restore(args.restore, trunk=args.trunk)

        # initialize trainer
        seqnn_trainer = trainer.Trainer(
            params_train, train_data, eval_data, args.out_dir, args.log_dir
        )

        # compile model
        seqnn_trainer.compile(seqnn_model)

    else:
        ########################################
        # multi GPU

        strategy = tf.distribute.MirroredStrategy()

        with strategy.scope():
            if not args.keras_fit:
                # distribute data
                for di in range(len(args.data_dirs)):
                    train_data[di].distribute(strategy)
                    eval_data[di].distribute(strategy)

            # initialize model
            seqnn_model = seqnn.SeqNN(params_model)

            # restore
            if args.restore:
                seqnn_model.restore(args.restore, args.trunk)

            # initialize trainer
            seqnn_trainer = trainer.Trainer(
                params_train,
                train_data,
                eval_data,
                args.out_dir,
                args.log_dir,
                strategy,
                params_train["num_gpu"],
                args.keras_fit,
            )

            # compile model
            seqnn_trainer.compile(seqnn_model)

    # train model
    if args.keras_fit:
        seqnn_trainer.fit_keras(seqnn_model)
    else:
        if len(args.data_dirs) == 1:
            seqnn_trainer.fit_tape(seqnn_model, resume=args.resume)
        else:
            seqnn_trainer.fit2(seqnn_model)


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
