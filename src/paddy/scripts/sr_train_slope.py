#!/usr/bin/env python3
"""
Train SlopeNN model for SloRice project.
"""
import argparse
import os
import sys
import time
import yaml
import numpy as np
import paddy.utils as utils
utils.set_gpu_device()

import shutil
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from tensorflow.keras import mixed_precision

from paddy import seqnn
from paddy import dataset
from paddy import trainer
from paddy import transfer
from paddy import utils

def parse_args():
    parser = argparse.ArgumentParser(description='Train SlopeNN model for SloRice')
    parser.add_argument(
        "-g",
        "--gpu",
        "--visible_device",
        nargs='+',
        type=str,
        default=None,
        help="GPU IDs to use (can specify multiple, e.g., -g 0 1 2)",
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
        default="sr_train_out",
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
        default=None,
        help="Resume training from checkpoint in args.out_dir [Default: %(default)s]",
    )
    parser.add_argument(
        "--restore",
        default=None,
        help="model trunk h5 file [Default: %(default)s]",
    )
    parser.add_argument(
        "--trunk",
        action="store_true",
        default=False,
        help="Restore only model trunk [Default: %(default)s]",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for model initialization [Default: %(default)s]",
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
    parser.add_argument(
        "--skip_train",
        action="store_true",
        default=False,
        help="report trainable params and skip training [Default: %(default)s]",
    )
    parser.add_argument('params_file', help="YAML file with model parameters")
    parser.add_argument('data_dirs', nargs='+', help="Train/valid/test data directorie(s)")
    
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()
    

    os.makedirs(args.out_dir, exist_ok=True)
    
    if args.params_file != "%s/params.yaml" % args.out_dir:
        shutil.copy(args.params_file, "%s/params.yaml" % args.out_dir)
    
    with open(args.params_file, 'r') as f:
        params = yaml.safe_load(f)
    
    params_model = params["model"]
    params_train = params["train"]
    transfer_mode = params["transfer"]["mode"]
    
    # Update num_gpu based on command line arguments
    if args.gpu and len(args.gpu) > 1:
        if params_train["num_gpu"] != len(args.gpu):
            params_train["num_gpu"] = len(args.gpu)
    
    seed = args.seed or params_train.get("seed", None)
    if seed is None:
        seed = random.randint(0, 1000000)  # random seed  
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    tf.config.experimental.enable_op_determinism()
    
    with open(f"{args.out_dir}/seed.txt", "w") as f:
        f.write(f"Random seed: {seed}\n")

    if args.keras_fit and len(args.data_dirs) > 1:
        print("Cannot use keras fit method with multi-genome training.")
        exit()

    params_model["trunk_model"] = args.restore
    train_data = []
    eval_data = []
    strand_pairs = []

    for data_dir in args.data_dirs:
        ################################################################################
        # data
        ################################################################################
        # Use SlopeDataset for SlopeNN data format
        data_train = dataset.SlopeDataset(
            data_dir,
            split_label="train",
            batch_size=params_train["batch_size"],
            shuffle_buffer=params_train["shuffle_buffer"],
            mode="train",
            repeat=True,
            tfr_pattern=args.tfr_train,
        )

        data_valid = dataset.SlopeDataset(
            data_dir,
            split_label="valid",
            batch_size=params_train["batch_size"],
            shuffle_buffer=params_train["shuffle_buffer"],
            mode="eval",
            repeat=False,
            tfr_pattern=args.tfr_eval,
        )

        # print dataset information
        print(f"Train dataset: {data_train.num_seqs} sequences")
        print(f"Valid dataset: {data_valid.num_seqs} sequences")
        print(f"Sequence length: {data_train.seq_length}")

        train_data.append(data_train)
        eval_data.append(data_valid)

    params_model["strand_pair"] = strand_pairs

    ################################################################################
    # model
    ################################################################################
    if args.mixed_precision:
        mixed_precision.set_global_policy("mixed_float16")

    if args.skip_train:
        exit(0)

    # Single GPU training
    if params_train.get("num_gpu", 1) == 1:
        # Initialize model
        params_model["verbose"] = True
        seqnn_model = seqnn.SlopeNN(params_model)

        # Restore model weights if specified
        if args.restore:
            seqnn_model.restore(args.restore, trunk=args.trunk)

        # Apply transfer learning weight freezing based on mode
        transfer.apply_transfer_mode(seqnn_model, transfer_mode)

        # Print model parameter information
        try:
            print("params in new head: %d" % transfer.param_count(seqnn_model.model))
            print(f"Model output shape: {seqnn_model.model.output_shape}")
        except Exception as e:
            print("Could not count parameters in head")

        seqnn_trainer = trainer.Trainer(params_train, train_data, eval_data, args.out_dir, args.log_dir)
        seqnn_trainer.compile(seqnn_model)
        
        # Start training for single GPU
        if args.keras_fit:
            seqnn_trainer.fit_keras(seqnn_model)
        else:
            if len(args.data_dirs) == 1:
                seqnn_trainer.fit_tape(seqnn_model, resume=args.resume)
            else:
                seqnn_trainer.fit2(seqnn_model)
    else:
        # Multi-GPU training
        strategy = tf.distribute.MirroredStrategy()

        with strategy.scope():
            if not args.keras_fit:
                for di in range(len(args.data_dirs)):
                    train_data[di].distribute(strategy)
                    eval_data[di].distribute(strategy)
                
            seqnn_model = seqnn.SlopeNN(params_model)

            if args.restore:
                seqnn_model.restore(args.restore, trunk=args.trunk)
            
            transfer.apply_transfer_mode(seqnn_model, transfer_mode)
            
            try:
                total_params = transfer.param_count(seqnn_model.model)
                trainable_params = sum([np.prod(w.shape) for w in seqnn_model.model.trainable_weights])
                print(f"Model: {total_params:,} total, {trainable_params:,} trainable parameters")
            except Exception as e:
                print(f"Could not count parameters: {e}")

            seqnn_trainer = trainer.Trainer(params_train, train_data, eval_data, args.out_dir, args.log_dir, strategy, params_train["num_gpu"], args.keras_fit)
            seqnn_trainer.compile(seqnn_model)


        if args.keras_fit:
            seqnn_trainer.fit_keras(seqnn_model)
        else:
            if len(args.data_dirs) == 1:
                seqnn_trainer.fit_tape(seqnn_model, resume=args.resume)
            else:
                seqnn_trainer.fit2(seqnn_model)
   


if __name__ == "__main__":
    main()
   
   