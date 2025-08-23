#!/usr/bin/env python

import argparse
import yaml
import os

import h5py
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras import mixed_precision

from paddy import bed
from paddy import dataset
from paddy import seqnn
from paddy import trainer
from paddy import metrics

"""
se_eval.py

Evaluate the accuracy of a trained sequence model on held-out sequences.
"""
    
def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained sequence model.")
    parser.add_argument(
        "-b",
        "--bedgraph_indexes",
        help="Comma-separated list of target indexes to write predictions/targets as bedgraph [Default: %(default)s]",
    )
    parser.add_argument(
        "--head",
        dest="head_i",
        default=0,
        type=int,
        help="Parameters head to evaluate [Default: %(default)s]",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        default="eval_out",
        help="Output directory for evaluation statistics [Default: %(default)s]",
    )
    parser.add_argument(
        "-sr",
        "--spearmanr",
        dest="spearmanr",
        default=False,
        action="store_true",
        help="Compute Spearman rank correlation [Default: %(default)s]",
    )
    parser.add_argument(
        "--rc",
        default=False,
        action="store_true",
        help="Average the fwd and rc predictions [Default: %(default)s]",
    )
    parser.add_argument(
        "--save",
        default=False,
        action="store_true",
        help="Save targets and predictions numpy arrays [Default: %(default)s]",
    )
    parser.add_argument(
        "--shifts",
        default="0",
        help="Ensemble prediction shifts [Default: %(default)s]",
    )
    parser.add_argument(
        "--step",
        default=1,
        type=int,
        help="Step across positions [Default: %(default)s]",
    )
    parser.add_argument(
        "-m",
        "--mixed-precision",
        default=False,
        action="store_true",
        help="use mixed precision for inference",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split label for eg TFR pattern [Default: %(default)s]",
    )
    parser.add_argument(
        "--tfr_pattern",
        default=None,
        help="TFR pattern string appended to data_dir/tfrecords for subsetting [Default: %(default)s]",
    )

    parser.add_argument("params_file", default=None, help="YAML file with model parameters")
    parser.add_argument("model_file", default=None, help="Trained model HDF5.")
    parser.add_argument("data_dir", help="Train/valid/test data directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # parse shifts to integers
    args.shifts = [int(shift) for shift in args.shifts.split(",")]

    #######################################################

    with open(args.params_file) as params_open:
        params = yaml.safe_load(params_open)
    params_model = params["model"]
    params_train = params["train"]
    params_model["verbose"] = False
    
    # construct eval data
    eval_data = dataset.SeqDataset(
        args.data_dir,
        split_label=args.split,
        batch_size=params_train["batch_size"],
        mode="eval",
        tfr_pattern=args.tfr_pattern,
        model_type=params_model["model_type"],
    )

    ###################
    # mixed precision #
    ###################
    if args.mixed_precision:
        mixed_precision.set_global_policy("mixed_float16")  # first set global policy
        seqnn_model = seqnn.SeqNN(params_model)  # then create model
        seqnn_model.restore(args.model_file, args.head_i)
        seqnn_model.append_activation()  # add additional activation to cast float16 output to float32
    else:
        # initialize model
        seqnn_model = seqnn.SeqNN(params_model)
        seqnn_model.restore(args.model_file, args.head_i)

    # Build ensemble with rc and shifts
    seqnn_model.build_ensemble(args.rc, args.shifts)

    #######################################################
    # evaluate
    loss_label = params_train.get("loss", "poisson").lower()
    spec_weight = params_train.get("spec_weight", 1)
    loss_fn = trainer.parse_loss(loss_label, spec_weight=spec_weight)

    # evaluate
    test_loss, test_metric1, test_metric2 = seqnn_model.evaluate(
        eval_data, loss_label=loss_label, loss_fn=loss_fn
    )

    # print summary statistics
    print("\nTest Loss:         %7.5f" % test_loss)

    if loss_label == "bce":
        print("Test AUROC:        %7.5f" % test_metric1.mean())
        print("Test AUPRC:        %7.5f" % test_metric2.mean())

        # write target-level statistics
        targets_acc_df = pd.DataFrame(
            {
                "index": np.arange(len(test_metric1)),
                "auroc": test_metric1,
                "auprc": test_metric2,
                "identifier": targets_df.identifier,
                "description": targets_df.description,
            }
        )

    else:
        print("Test PearsonR:     %7.5f" % test_metric1.mean())
        print("Test R2:           %7.5f" % test_metric2.mean())
        
        metrics_df = pd.DataFrame(
            {
                "identifier": [args.tfr_pattern],
                "test_loss": [test_loss],
                "test_PearsonR": [test_metric1.mean()],
                "test_R2": [test_metric2.mean()],
            }
        )
        metrics_df.to_csv(os.path.join(args.out_dir, "metrics.tsv"), sep="\t", index=False, float_format="%.5f")
        # write target-level statistics 
        targets_acc_df = pd.DataFrame(
            {
                "index": np.arange(len(test_metric1)),
                "pearsonr": test_metric1,
                "r2": test_metric2,
            }
        )

    targets_acc_df.to_csv(
        "%s/acc.tsv" % args.out_dir, sep="\t", index=False, float_format="%.5f"
    )

    #######################################################
    # if we want to save/spearman, predict again

    if args.save or args.spearmanr:
        # compute predictions
        print("\nComputing predictions for detailed evaluation...")
        test_preds = seqnn_model.predict(
            eval_data, stream=True, step=args.step, dtype="float16"
        )

        # read targets
        test_targets = eval_data.numpy(return_inputs=False, step=args.step)

        if args.spearmanr:
            # compute target spearmanr
            test_spearmanr = []
            for ti in tqdm(range(test_preds.shape[-1])):
                test_preds_flat = test_preds[..., ti].flatten()
                test_targets_flat = test_targets[..., ti].flatten()
                spear_ti = spearmanr(test_targets_flat, test_preds_flat)[0]
                test_spearmanr.append(spear_ti)

            # write target-level statistics
            targets_acc_df = pd.DataFrame(
                {
                    "index": np.arange(len(test_metric1)),
                    "pearsonr": test_metric1,
                    "spearmanr": test_spearmanr,
                    "r2": test_metric2,
                }
            )

            targets_acc_df.to_csv(
                "%s/acc.tsv" % args.out_dir, sep="\t", index=False, float_format="%.5f"
            )

    if args.save:
        with h5py.File("%s/preds.h5" % args.out_dir, "w") as preds_h5:
            preds_h5.create_dataset("preds", data=test_preds)
        with h5py.File("%s/targets.h5" % args.out_dir, "w") as targets_h5:
            targets_h5.create_dataset("targets", data=test_targets)

        if args.bedgraph_indexes is not None:
            print(f"Not checked this function yet, please remove --bedgraph_indexes")
            exit()
            bedgraph_indexes = [int(ti) for ti in args.bedgraph_indexes.split(",")]
            bedg_out = "%s/bedgraph" % args.out_dir
            bed.write_bedgraph(
                test_preds,
                test_targets,
                args.data_dir,
                bedg_out,
                args.split,
                bedgraph_indexes,
            )


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main() 