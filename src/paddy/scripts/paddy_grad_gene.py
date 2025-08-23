#!/usr/bin/env python

from __future__ import print_function

from optparse import OptionParser

import gc
import yaml
import os
import time

import h5py
import numpy as np
import pandas as pd
import pysam
import tensorflow as tf

from paddy import dna as dna_io
from paddy import gene as pgene
from paddy import seqnn

"""
paddy_grad_gene.py

Perform a gradient saliency analysis for genes specified in a GTF file.
Supports both 2d_to_1d (1D output) and 2d_to_2d (2D output) model types.
Simplified for single model usage.
"""

################################################################################
# main
################################################################################
def main():
    usage = "usage: %prog [options] --model_path <model_file> <params> <gene_gtf>"
    parser = OptionParser(usage)
    parser.add_option(
        "--fa",
        dest="genome_fasta",
        default=None,
        help="Genome FASTA for sequences [Default: %default]",
    )
    parser.add_option(
        "-o",
        dest="out_file",
        default="grad_out.h5",
        help="Output h5 file, containing all tissues' gradients and input*gradients [Default: %default]",
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
    # parser.add_option(
    #     "--track_transform",
    #     dest="track_transform",
    #     default=0.75,
    #     type="float",
    #     help="Target transform exponent [Default: %default]",
    # )
    # parser.add_option(
    #     "--clip_soft",
    #     dest="clip_soft",
    #     default=384.0,
    #     type="float",
    #     help="Model clip_soft setting [Default: %default]",
    # )
    parser.add_option(
        "--get_preds",
        dest="get_preds",
        default=False,
        action="store_true",
        help="Store scalar predictions in addition to their gradients [Default: %default]",
    )
    parser.add_option(
        "--no_untransform",
        dest="no_untransform",
        default=False,
        action="store_true",
        help="Skip untransformation of predictions [Default: %default]",
    )
    parser.add_option(
        "--atg",
        dest="center_on_atg",
        default=False,
        action="store_true",
        help="Center sequence window on ATG start codon instead of gene midpoint [Default: %default]",
    )
    parser.add_option(
        "--batch_size",
        dest="batch_size",
        default=1,
        type="int",
        help="Batch size for gradient computation [Default: %default]",
    )

    (options, args) = parser.parse_args()

    if len(args) == 2:
        # single worker
        params_file = args[0]
        genes_gtf_file = args[1]
    else:
        parser.error("Must provide parameter file and GTF file")

    # Model path is required
    if options.model_path is None:
        parser.error("--model_path is required. Please specify the path to your model file.")
 
    model_path = options.model_path
    if not os.path.isfile(model_path):
        parser.error(f"Model file not found: {model_path}")
    
    options.shifts = [int(shift) for shift in options.shifts.split(",")]

    #################################################################
    # read parameters

    # read model parameters
    with open(params_file) as params_open:
        params = yaml.safe_load(params_open)
    params_model = params["model"]
    seq_len = params_model["seq_length"]
    model_type = params_model["model_type"]

    if model_type != "2d_to_1d":
        raise ValueError(f"Model type {model_type} not supported, you can use borzoi_satg_gene.py instead")
    
    #################################################################
    # load model

    seqnn_model = seqnn.SeqNN(params_model)
    seqnn_model.restore(model_path, options.head_i)

    #################################################################
    # read genes

    # parse GTF
    transcriptome = pgene.Transcriptome(genes_gtf_file)

    # order valid genes
    genome_open = pysam.Fastafile(options.genome_fasta)
    gene_list = sorted(transcriptome.genes.keys())
    num_genes = len(gene_list)

    #################################################################
    # setup output

    # choose gene sequences
    genes_chr = []
    genes_start = []
    genes_end = []
    genes_strand = []
    for gene_id in gene_list:
        gene = transcriptome.genes[gene_id]
        genes_chr.append(gene.chrom)
        genes_strand.append(gene.strand)

        if options.center_on_atg:
            # Center on ATG start codon (CDS start)
            if gene.strand == "+":
                # For forward strand, ATG is at the start of the first CDS
                atg_pos = gene.cds_start()
            else:
                # For reverse strand, ATG is at the end of the last CDS (reverse complement)
                atg_pos = gene.cds_end()
            
            # Center the sequence window on ATG position
            gene_start = max(0, atg_pos - seq_len // 2)
            gene_end = gene_start + seq_len
        else:
            # Original behavior: center on gene midpoint
            gene_midpoint = gene.midpoint()
            gene_start = max(0, gene_midpoint - seq_len // 2)
            gene_end = gene_start + seq_len
            
        genes_start.append(gene_start)
        genes_end.append(gene_end)

    #################################################################
    # predict scores, write output

    buffer_size = 1024
    # Set num_targets based on model type
    num_targets = params_model["num_targets"]

    print("n genes = " + str(len(genes_chr)))
    print(f"Computing gradients for {num_targets} tissues")
    if options.center_on_atg:
        print("Centering sequences on ATG start codon (CDS start)")
        # Check if CDS information is available
        genes_with_cds = 0
        for gene_id in gene_list:
            gene = transcriptome.genes[gene_id]
            if hasattr(gene, 'cds_intervals') and len(gene.get_cds()) > 0:
                genes_with_cds += 1
        
        if genes_with_cds == 0:
            print("WARNING: No CDS information found in GTF file.")
            print("         Falling back to gene start/end positions.")
            print("         For true ATG centering, ensure your GTF contains CDS features.")
        elif genes_with_cds != len(gene_list):
            print(f"WARNING: {len(gene_list) - genes_with_cds} genes have no CDS information")
        else:
            print(f"Found CDS information for {genes_with_cds}/{len(gene_list)} genes")
    else:
        print("Centering sequences on gene midpoint")

    # Initialize HDF5
    scores_h5_file = options.out_file
    if os.path.isfile(scores_h5_file):
        os.remove(scores_h5_file)
    scores_h5 = h5py.File(scores_h5_file, "w")
    
    # Create HDF5 datasets - only store gradients for all tissues
    scores_h5.create_dataset("seqs", dtype="bool", shape=(num_genes, seq_len, 4))
    scores_h5.create_dataset(
        "grads", dtype="float16", shape=(num_genes, seq_len, 4, num_targets)
    )
    scores_h5.create_dataset(
        "grads_saliency", dtype="float16", shape=(num_genes, seq_len, 4, num_targets)
    )
    if options.get_preds:
        scores_h5.create_dataset(
            "preds", dtype="float32", shape=(num_genes, num_targets)
        )
    scores_h5.create_dataset("gene", data=np.array(gene_list, dtype="S"))
    scores_h5.create_dataset("chr", data=np.array(genes_chr, dtype="S"))
    scores_h5.create_dataset("start", data=np.array(genes_start))
    scores_h5.create_dataset("end", data=np.array(genes_end))
    scores_h5.create_dataset("strand", data=np.array(genes_strand, dtype="S"))
    
    # Get predictions if requested
    if options.get_preds:
        print(" - (prediction) - ", flush=True)
        
        for shift in options.shifts:
            print("Processing shift %d" % shift, flush=True)

            for rev_comp in [False, True] if options.rc else [False]:
                if options.rc:
                    print("Fwd/rev = %s" % ("fwd" if not rev_comp else "rev"), flush=True)

                seq_1hots = []

                for gi, gene_id in enumerate(gene_list):
                    if gi % 500 == 0:
                        print("Processing %d, %s" % (gi, gene_id), flush=True)

                    # make sequence
                    seq_1hot = make_seq_1hot(
                        genome_open, genes_chr[gi], genes_start[gi], genes_end[gi], seq_len
                    )
                    seq_1hot = dna_io.hot1_augment(seq_1hot, shift=shift)

                    if rev_comp:
                        seq_1hot = dna_io.hot1_rc(seq_1hot)

                    seq_1hots.append(seq_1hot[None, ...])

                    if gi == len(gene_list) - 1 or len(seq_1hots) >= buffer_size:
                        # concat sequences
                        seq_1hots = np.concatenate(seq_1hots, axis=0)

                        # batch call predictions
                        preds = seqnn_model(seq_1hots)

                        # save predictions for all tissues
                        for gii in range(len(seq_1hots)):
                            h5_gi = (gi // buffer_size) * buffer_size + gii
                            scores_h5["preds"][h5_gi] += preds[gii, :] / float(
                                (len(options.shifts) * (2 if options.rc else 1))
                            )

                        seq_1hots = []
                        gc.collect()

    # Compute gradients
    print(" - (gradients) - ", flush=True)

    for shift in options.shifts:
        print("Processing shift %d" % shift, flush=True)

        for rev_comp in [False, True] if options.rc else [False]:
            if options.rc:
                print("Fwd/rev = %s" % ("fwd" if not rev_comp else "rev"), flush=True)

            seq_1hots = []

            for gi, gene_id in enumerate(gene_list):
                if gi % 500 == 0:
                    print("Processing %d, %s" % (gi, gene_id), flush=True)

                # make sequence
                seq_1hot = make_seq_1hot(
                    genome_open, genes_chr[gi], genes_start[gi], genes_end[gi], seq_len
                )
                seq_1hot = dna_io.hot1_augment(seq_1hot, shift=shift)

                if rev_comp:
                    seq_1hot = dna_io.hot1_rc(seq_1hot)

                seq_1hots.append(seq_1hot[None, ...])

                if gi == len(gene_list) - 1 or len(seq_1hots) >= buffer_size:
                    # concat sequences
                    seq_1hots = np.concatenate(seq_1hots, axis=0)

                    # batch call gradient computation for all tissues
                    grads = seqnn_model.gradients(
                        seq_1hots,
                        head_i=options.head_i,
                        target_slice=np.array([[i] for i in range(num_targets)]),
                        pos_slice=np.array([[0]]),
                        chunk_size=buffer_size,
                        batch_size=options.batch_size,
                        track_scale=options.track_scale,
                        # track_transform=options.track_transform,
                        # clip_soft=options.clip_soft,
                        pseudo_count=options.pseudo,
                        no_untransform=options.no_untransform,
                        subtract_avg=True,
                        input_gate=False,
                        dtype="float16",
                    )
                    
                    # batch call input gradients (input × gradients) for all tissues
                    input_grads = seqnn_model.gradients(
                        seq_1hots,
                        head_i=options.head_i,
                        target_slice=np.array([[i] for i in range(num_targets)]),
                        pos_slice=np.array([[0]]),
                        chunk_size=buffer_size,
                        batch_size=options.batch_size,
                        track_scale=options.track_scale,
                        # track_transform=options.track_transform,
                        # clip_soft=options.clip_soft,
                        pseudo_count=options.pseudo,
                        no_untransform=options.no_untransform,
                        subtract_avg=True,
                        input_gate=True,
                        dtype="float16",
                    )

                    # undo augmentations and save gradients and input gradients
                    for gii in range(len(seq_1hots)):
                        # Extract gradients for all tissues
                        # grads shape: [num_genes, seq_length, seq_depth, num_tissues]
                        grad = unaugment_grads(
                            grads[gii, :, :, :],  # [seq_length, seq_depth, num_tissues]
                            fwdrc=(not rev_comp),
                            shift=shift,
                        )
                        
                        # Extract input gradients for all tissues
                        input_grad = unaugment_grads(
                            input_grads[gii, :, :, :],  # [seq_length, seq_depth, num_tissues]
                            fwdrc=(not rev_comp),
                            shift=shift,
                        )

                        h5_gi = (gi // buffer_size) * buffer_size + gii
                        scores_h5["grads"][h5_gi] += grad
                        scores_h5["grads_saliency"][h5_gi] += input_grad

                    seq_1hots = []
                    gc.collect()

    # Save sequences and normalize gradients and input gradients
    print(" - (saving sequences and normalizing) - ", flush=True)
    for gi, gene_id in enumerate(gene_list):
        if gi % 500 == 0:
            print("Processing %d, %s" % (gi, gene_id), flush=True)
            
        # re-make original sequence
        seq_1hot = make_seq_1hot(
            genome_open, genes_chr[gi], genes_start[gi], genes_end[gi], seq_len
        )

        # write sequence to HDF5
        scores_h5["seqs"][gi] = seq_1hot
        
        # normalize gradients and input gradients
        normalization_factor = float(len(options.shifts) * (2 if options.rc else 1))
        scores_h5["grads"][gi] /= normalization_factor
        scores_h5["grads_saliency"][gi] /= normalization_factor

    # Close files
    scores_h5.close()
    genome_open.close()
    gc.collect()


def unaugment_grads(grads, fwdrc=False, shift=0):
    """Undo sequence augmentation."""
    # Handle different gradient array shapes
    original_shape = grads.shape
    
    if len(grads.shape) == 2:
        # 2D case: (seq_len, 4) - add dummy dimensions for compatibility
        grads = grads[:, :, np.newaxis, np.newaxis]
        squeeze_output = True
    elif len(grads.shape) == 3:
        # 3D case: (seq_len, seq_depth, num_tissues) - no change needed
        # This is our expected 4D case for 2d_to_1d models
        squeeze_output = False
    elif len(grads.shape) == 4:
        # 4D case: (seq_len, seq_depth, num_tissues) - no change needed
        squeeze_output = False
    else:
        raise ValueError(f"Unexpected gradient shape: {grads.shape}")
    
        # reverse complement
    if not fwdrc:
        # reverse along sequence dimension
        if len(grads.shape) == 4:
            grads = grads[::-1, :, :, :]
        else:
            grads = grads[::-1, :, :]

        # Handle different dimensions for swapping
        if len(grads.shape) == 4:
            # 4D case: [seq_len, seq_depth, num_tissues, extra_dim]
            # swap A and T (indices 0 and 3)
            grads[:, [0, 3], :, :] = grads[:, [3, 0], :, :]
            # swap C and G (indices 1 and 2)  
            grads[:, [1, 2], :, :] = grads[:, [2, 1], :, :]
        else:
            # 3D case: [seq_len, seq_depth, num_tissues]
            # swap A and T (indices 0 and 3)
            grads[:, [0, 3], :] = grads[:, [3, 0], :]
            # swap C and G (indices 1 and 2)  
            grads[:, [1, 2], :] = grads[:, [2, 1], :]

    # undo shift
    if len(grads.shape) == 4:
        # 4D case
        if shift < 0:
            # shift sequence right
            grads[-shift:, :, :, :] = grads[:shift, :, :, :]
            # fill in left unknowns
            grads[:-shift, :, :, :] = 0
        elif shift > 0:
            # shift sequence left
            grads[:-shift, :, :, :] = grads[shift:, :, :, :]
            # fill in right unknowns
            grads[-shift:, :, :, :] = 0
    else:
        # 3D case
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
    if squeeze_output:
        if len(grads.shape) == 4 and grads.shape[-1] == 1:
            grads = grads[:, :, :, 0]  # Remove last dimension
        if len(grads.shape) == 4 and grads.shape[-2] == 1:
            grads = grads[:, :, 0, :]  # Remove second-to-last dimension

    return grads


def make_seq_1hot(genome_open, chrm, start, end, seq_len):
    if start < 0:
        raise ValueError(f"For {chrm}, start: {start}, end: {end} is out of range")
        # seq_dna = "N" * (-start) + genome_open.fetch(chrm, 0, end)
    else:
        seq_dna = genome_open.fetch(chrm, start, end)

    # extend to full length
    if len(seq_dna) < seq_len:
        seq_dna += "N" * (seq_len - len(seq_dna))

    seq_1hot = dna_io.dna_1hot(seq_dna)
    return seq_1hot




################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    import sys
    # debugging
    if len(sys.argv) == 1:
        sys.argv += [
        "--model_path", "/data/xhhuanglab/gulei/projects/paddy/seq2exp/best_model_dirs/34P_23tracks_34P_TrunkFrozen_PaddyHead_best_model_dir/seed100_model_best.h5",
        "--fa", "/data/xhhuanglab/gulei/projects/paddy/motif_pipeline/NumChr.Rice_MSUv7.fa",
        "--rc",
        "--atg",
        "/data/xhhuanglab/gulei/projects/paddy/seq2exp/transfer_CE_PaddyHead.yaml",
        "-o", "/data/xhhuanglab/gulei/projects/paddy/motif_pipeline/tmp/tissues.h5",
        "/data/xhhuanglab/gulei/projects/paddy/motif_pipeline/tmp/tmp.gff3"
        ]
    main()