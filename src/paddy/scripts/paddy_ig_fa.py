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
paddy_ig_fa.py
Perform Integrated Gradients (IG) attribution analysis for sequences in a FASTA file.
Supports 2d_to_1d (1D output) model types.

Reference: Sundararajan et al. (2017) "Axiomatic Attribution for Deep Networks"
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
        default="ig_out.h5",
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
        default=8,
        type="int",
        help="Batch size for gradient computation [Default: %default]",
    )
    parser.add_option(
        "--num_steps",
        dest="num_steps",
        default=50,
        type="int",
        help="Number of interpolation steps for Integrated Gradients [Default: %default]",
    )
    parser.add_option(
        "--baseline",
        dest="baseline_type",
        default="zero",
        type="str",
        help="Baseline type of Integrated Gradients: 'zero' for all zeros, 'uniform' for uniform distribution, 'gc' for GC-content based [Default: %default]",
    )
    parser.add_option(
        "--gc_content",
        dest="gc_content",
        default=0.435408,
        type="float",
        help="GC content for 'gc' baseline type (rice Nipponbare default: 0.435408) [Default: %default]",
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
    parser.add_option(
        "--tissue_indices",
        dest="tissue_indices",
        default=None,
        type="str",
        help="Comma-separated tissue indices to compute (e.g., '0,1,2,3,4' for first 5 tissues). If None, compute all tissues [Default: %default]",
    )
    parser.add_option(
        "--tissue_slab",
        dest="tissue_slab",
        default=1,
        type="int",
        help="Number of tissues to process at once for memory efficiency. tissue_slab=1 uses minimal memory but is slower; larger values are faster but use more memory [Default: %default]",
    )
    parser.add_option(
        "--importance_scores",
        dest="importance_scores",
        default=False,
        action="store_true",
        help="Compute global importance scores by averaging over all tissues before computing gradients. This gives sequence-level importance independent of specific tissues [Default: %default]",
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
    
    # Parse tissue indices
    if options.tissue_indices is not None:
        tissue_indices = [int(idx) for idx in options.tissue_indices.split(",")]
    else:
        tissue_indices = None
    
    # Read model parameters
    with open(params_file) as params_open:
        params = yaml.safe_load(params_open)
    params_model = params["model"]
    seq_len = params_model["seq_length"]
    model_type = params_model.get("model_type", "2d_to_1d")
    if model_type != "2d_to_1d":
        raise ValueError(f"Model type {model_type} not supported for this script. Have not implemented integrated gradients for 2d_to_2d models yet.")

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
    
    # Determine output shape based on tissue_indices and importance_scores mode
    if options.importance_scores:
        # Global importance mode: compute gradient w.r.t. mean of all tissues
        print(f"Computing global importance scores (average over all {num_targets} tissues)")
        num_output_targets = 1
        tissue_indices = list(range(num_targets))  # Use all tissues for averaging
        output_label = "importance"
    else:
        # Per-tissue mode
        if tissue_indices is not None:
            num_output_targets = len(tissue_indices)
            print(f"Computing gradients for {num_output_targets} selected tissues: {tissue_indices}")
        else:
            num_output_targets = num_targets
            tissue_indices = list(range(num_targets))
            print(f"Computing gradients for all {num_targets} tissues")
        output_label = "igs"
    
    scores_h5.create_dataset("seqs", dtype="bool", shape=(num_seqs, seq_len, 4))
    scores_h5.create_dataset(
        output_label, dtype="float16", shape=(num_seqs, seq_len, 4, num_output_targets)
    )
    scores_h5.create_dataset("fasta_headers", data=np.array(fasta_headers, dtype="S"))
    if not options.importance_scores:
        scores_h5.create_dataset("tissue_indices", data=np.array(tissue_indices, dtype="int32"))
    
    # Create baseline
    if options.baseline_type == "zero":
        baseline = None  # Will use all zeros in integrated_gradients
    elif options.baseline_type == "uniform":
        baseline = np.ones((1, seq_len, 4), dtype="float32") * 0.25
    elif options.baseline_type == "gc":
        # GC-content based baseline: A, C, G, T frequencies
        gc_content = options.gc_content
        at_freq = (1.0 - gc_content) / 2.0  # A and T frequency
        gc_freq = gc_content / 2.0  # C and G frequency
        baseline = np.ones((1, seq_len, 4), dtype="float32")
        baseline[0, :, 0] = at_freq  # A
        baseline[0, :, 1] = gc_freq  # C
        baseline[0, :, 2] = gc_freq  # G
        baseline[0, :, 3] = at_freq  # T
    else:
        raise ValueError(f"Unknown baseline type: {options.baseline_type}")
    
    # Compute integrated gradients
    print(f"Computing Integrated Gradients for {num_seqs} sequences...")
    print(f"Using {options.num_steps} interpolation steps")
    print(f"Baseline type: {options.baseline_type}")
    if options.baseline_type == "gc":
        print(f"GC content: {options.gc_content:.6f} (A/T: {(1-options.gc_content)/2:.6f}, C/G: {options.gc_content/2:.6f})")
    print(f"Memory optimization: tissue_slab={options.tissue_slab} (processing {options.tissue_slab} tissue(s) at a time)")
    
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
                if len(seq_dna) != seq_len:
                    exit(f"Sequence {header} (seq_len={len(seq_dna)}), must be equal to model input length {seq_len}")

                seq_1hot = dna_io.dna_1hot(seq_dna)
                seq_1hot = dna_io.hot1_augment(seq_1hot, shift=shift)
                if rev_comp:
                    seq_1hot = dna_io.hot1_rc(seq_1hot)
                
                seq_1hots.append(seq_1hot[None, ...])
                header_batch_indices.append(si)

                if si == num_seqs - 1 or len(seq_1hots) >= buffer_size:
                    seq_1hots = np.concatenate(seq_1hots, axis=0)
                    
                    # Prepare baseline for this batch
                    if baseline is None:
                        batch_baseline = None
                    else:
                        # Augment baseline same as sequences
                        batch_baseline = dna_io.hot1_augment(baseline[0], shift=shift)
                        if rev_comp:
                            batch_baseline = dna_io.hot1_rc(batch_baseline)
                        batch_baseline = np.tile(batch_baseline[None, ...], (len(seq_1hots), 1, 1))
                    
                    # Compute integrated gradients
                    # Note: IG inherently includes input information, no need for separate input_gate
                    if options.importance_scores:
                        # Global importance: compute gradient w.r.t. mean of all tissues
                        igs = seqnn_model.integrated_gradients(
                            seq_1hots,
                            baseline=batch_baseline,
                            num_steps=options.num_steps,
                            head_i=options.head_i,
                            target_slice=np.array([[i] for i in tissue_indices]),
                            pos_slice=np.array([[0]]),
                            chunk_size=buffer_size,
                            batch_size=options.batch_size,
                            track_scale=options.track_scale,
                            pseudo_count=options.pseudo,
                            no_untransform=options.no_untransform,
                            subtract_avg=True,
                            tissue_slab=options.tissue_slab,
                            dtype="float16",
                            use_mean_target=True,  # Average over all tissues before computing gradient
                        )
                    else:
                        # Per-tissue mode: compute gradient for each tissue separately
                        igs = seqnn_model.integrated_gradients(
                            seq_1hots,
                            baseline=batch_baseline,
                            num_steps=options.num_steps,
                            head_i=options.head_i,
                            target_slice=np.array([[i] for i in tissue_indices]),
                            pos_slice=np.array([[0]]),
                            chunk_size=buffer_size,
                            batch_size=options.batch_size,
                            track_scale=options.track_scale,
                            pseudo_count=options.pseudo,
                            no_untransform=options.no_untransform,
                            subtract_avg=True,
                            tissue_slab=options.tissue_slab,
                            dtype="float16",
                        )
                    
                    for i in range(len(seq_1hots)):
                        h5_si = header_batch_indices[i]
                        ig = unaugment_grads(igs[i], fwdrc=(not rev_comp), shift=shift)
                        scores_h5[output_label][h5_si] += ig
                    
                    seq_1hots = []
                    header_batch_indices = []
                    gc.collect()

    # Save sequences and normalize integrated gradients
    print("Saving sequences and normalizing integrated gradients...")
    for si, header in enumerate(fasta_headers):
        seq_dna = fasta_open.fetch(header)
        if len(seq_dna) != seq_len:
            exit(f"Sequence {header} (seq_len={len(seq_dna)}), must be equal to model input length {seq_len}")

        seq_1hot = dna_io.dna_1hot(seq_dna)
        scores_h5["seqs"][si] = seq_1hot
        norm_factor = float(len(options.shifts) * (2 if options.rc else 1))
        if norm_factor > 0:
            scores_h5[output_label][si] /= norm_factor

    scores_h5.close()
    fasta_open.close()
    print("Done.")

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

