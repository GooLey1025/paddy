#!/usr/bin/env python
from __future__ import print_function
from optparse import OptionParser
import gc
import yaml
import os
import sys
import h5py
import numpy as np
import pysam
import tensorflow as tf

from paddy import dna as dna_io
from paddy import seqnn

"""
paddy_grad_fa.py
Perform a gradient saliency analysis for sequences in a FASTA file.
Supports 2d_to_1d (1D output) model types.
"""
################################################################################
# main
################################################################################
def main():
    usage = "usage: %prog [options] <params_file> <fasta_file>"
    parser = OptionParser(usage)
    parser.add_option(
        "-o",
        dest="out_file",
        default="grad_out.h5",
        help="Output h5 file [Default: %default]",
    )
    parser.add_option(
        "--rc",
        dest="rc",
        default=False,
        action="store_true",
        help="Ensemble forward and reverse complement predictions [Default: %default]",
    )
    parser.add_option(
        "--model_path",
        dest="model_path",
        default=None,
        type="str",
        help="Direct path to model file (e.g., model_best.h5) [Default: %default]",
    )
    parser.add_option(
        "--head",
        dest="head_i",
        default=0,
        type="int",
        help="Model head index [Default: %default]",
    )
    parser.add_option(
        "--shifts",
        dest="shifts",
        default="0",
        type="str",
        help="Ensemble prediction shifts [Default: %default]",
    )

    parser.add_option(
        "--batch_size",
        dest="batch_size",
        default=1,
        type="int",
        help="Batch size for gradient computation [Default: %default]",
    )
    parser.add_option(
        "--pseudo",
        dest="pseudo",
        default=0.0,
        type="float",
        help="Constant pseudo count [Default: %default]",
    )
    parser.add_option(
        "--track_scale",
        dest="track_scale",
        default=1.0,
        type="float",
        help="Target transform scale [Default: %default]",
    )
    parser.add_option(
        "--no_untransform",
        dest="no_untransform",
        default=False,
        action="store_true",
        help="Skip untransformation of predictions [Default: %default]",
    )
    (options, args) = parser.parse_args()
    if len(args) == 2:
        params_file = args[0]
        fasta_file = args[1]
    else:
        parser.error("Must provide parameter file and FASTA file")
    if options.model_path is None:
        parser.error("--model_path is required.")
    if not os.path.isfile(options.model_path):
        parser.error(f"Model file not found: {options.model_path}")
    options.shifts = [int(shift) for shift in options.shifts.split(",")]
    # Read model parameters
    with open(params_file) as params_open:
        params = yaml.safe_load(params_open)
    params_model = params["model"]
    seq_len = params_model["seq_length"]
    model_type = params_model.get("model_type", "2d_to_1d")
    if model_type != "2d_to_1d":
        raise ValueError(f"Model type {model_type} not supported for this script. Have not implemented gradients for 2d_to_2d models yet.")

    # Load model
    seqnn_model = seqnn.SeqNN(params_model)
    seqnn_model.restore(options.model_path, options.head_i)
    # Read sequences from FASTA
    fasta_open = pysam.Fastafile(fasta_file)
    fasta_headers = list(fasta_open.references)
    num_seqs = len(fasta_headers)
    # Setup output HDF5
    scores_h5_file = options.out_file
    if os.path.isfile(scores_h5_file):
        os.remove(scores_h5_file)
    scores_h5 = h5py.File(scores_h5_file, "w")
    num_targets = params_model["num_targets"]
    scores_h5.create_dataset("seqs", dtype="bool", shape=(num_seqs, seq_len, 4))
    scores_h5.create_dataset(
        "grads", dtype="float16", shape=(num_seqs, seq_len, 4, num_targets)
    )
    scores_h5.create_dataset(
        "grads_saliency", dtype="float16", shape=(num_seqs, seq_len, 4, num_targets)
    )
    scores_h5.create_dataset("fasta_headers", data=np.array(fasta_headers, dtype="S"))
    # Compute gradients
    print(f"Computing gradients for {num_seqs} sequences...")
    buffer_size = 1000
    for shift in options.shifts:
        print(f"Processing shift {shift}")
        for rev_comp in [False, True] if options.rc else [False]:
            if options.rc:
                print(f"Fwd/rev = {'rev' if rev_comp else 'fwd'}")
            
            seq_1hots = []
            header_batch_indices = []
            for si, header in enumerate(fasta_headers):
                seq_dna = fasta_open.fetch(header)
                
                # Take center of sequence if it's longer than model input
                if len(seq_dna) > seq_len:
                    print(f"Warning: Sequence {header} is too long. Taking center of sequence.", file=sys.stderr)
                    center_pos = len(seq_dna) // 2
                    start = center_pos - seq_len // 2
                    end = start + seq_len
                    seq_dna = seq_dna[start:end]

                seq_1hot = make_seq_1hot(seq_dna, seq_len)
                seq_1hot = dna_io.hot1_augment(seq_1hot, shift=shift)
                if rev_comp:
                    seq_1hot = dna_io.hot1_rc(seq_1hot)
                
                seq_1hots.append(seq_1hot[None, ...])
                header_batch_indices.append(si)

                if si == num_seqs - 1 or len(seq_1hots) >= buffer_size:
                    seq_1hots = np.concatenate(seq_1hots, axis=0)
                    
                    grads = seqnn_model.gradients(
                        seq_1hots,
                        head_i=options.head_i,
                        target_slice=np.array([[i] for i in range(num_targets)]),
                        pos_slice=np.array([[0]]),
                        chunk_size=buffer_size,
                        batch_size=options.batch_size,
                        track_scale=options.track_scale,
                        pseudo_count=options.pseudo,
                        no_untransform=options.no_untransform,
                        subtract_avg=True,
                        input_gate=False,
                        dtype="float16",
                    )
                    input_grads = seqnn_model.gradients(
                        seq_1hots,
                        head_i=options.head_i,
                        target_slice=np.array([[i] for i in range(num_targets)]),
                        pos_slice=np.array([[0]]),
                        chunk_size=buffer_size,
                        batch_size=options.batch_size,
                        track_scale=options.track_scale,
                        pseudo_count=options.pseudo,
                        no_untransform=options.no_untransform,
                        subtract_avg=True,
                        input_gate=True,
                        dtype="float16",
                    )
                    
                    for i in range(len(seq_1hots)):
                        h5_si = header_batch_indices[i]
                        grad = unaugment_grads(grads[i], fwdrc=(not rev_comp), shift=shift)
                        input_grad = unaugment_grads(input_grads[i], fwdrc=(not rev_comp), shift=shift)
                        scores_h5["grads"][h5_si] += grad
                        scores_h5["grads_saliency"][h5_si] += input_grad
                    
                    seq_1hots = []
                    header_batch_indices = []
                    gc.collect()

    # Save sequences and normalize gradients
    print("Saving sequences and normalizing gradients...")
    for si, header in enumerate(fasta_headers):
        seq_dna = fasta_open.fetch(header)
        if len(seq_dna) > seq_len:
            center_pos = len(seq_dna) // 2
            start = center_pos - seq_len // 2
            end = start + seq_len
            seq_dna = seq_dna[start:end]

        seq_1hot = make_seq_1hot(seq_dna, seq_len)
        scores_h5["seqs"][si] = seq_1hot
        norm_factor = float(len(options.shifts) * (2 if options.rc else 1))
        if norm_factor > 0:
            scores_h5["grads"][si] /= norm_factor
            scores_h5["grads_saliency"][si] /= norm_factor

    scores_h5.close()
    fasta_open.close()
    print("Done.")

def make_seq_1hot(seq_dna, seq_len):
    if len(seq_dna) < seq_len:
        seq_dna += 'N' * (seq_len - len(seq_dna))
    return dna_io.dna_1hot(seq_dna)

def unaugment_grads(grads, fwdrc=True, shift=0):
    """Undo sequence augmentation."""
    # Handle different gradient array shapes
    original_shape = grads.shape
    
    if len(grads.shape) == 2:
        # 2D case: (seq_len, 4) - add dummy dimensions for compatibility
        grads = grads[:, :, np.newaxis]
        squeeze_output = True
    elif len(grads.shape) == 3:
        # 3D case: (seq_len, seq_depth, num_tissues) - this is expected
        squeeze_output = False
    else:
        raise ValueError(f"Unexpected gradient shape: {grads.shape}")
    
    # reverse complement
    if not fwdrc:
        # reverse along sequence dimension
        grads = grads[::-1, :, :]
        # swap A and T (indices 0 and 3)
        grads[:, [0, 3], :] = grads[:, [3, 0], :]
        # swap C and G (indices 1 and 2)
        grads[:, [1, 2], :] = grads[:, [2, 1], :]
    
    # undo shift
    if shift < 0:
        # shift sequence right
        grads[-shift:, :, :] = grads[:shift, :, :]
        # fill in left unknowns
        grads[:-shift, :, :] = 0
    elif shift > 0:
        # shift sequence left
        grads[:-shift, :, :] = grads[shift:, :, :]
        # fill in right unknowns
        grads[-shift:, :, :] = 0
    
    # Remove dummy dimensions if we added them
    if squeeze_output and len(grads.shape) == 3 and grads.shape[-1] == 1:
        grads = grads[:, :, 0]
            
    return grads
################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
